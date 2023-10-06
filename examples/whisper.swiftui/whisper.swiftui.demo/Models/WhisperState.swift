import AVFoundation
import Combine
import Foundation
import SwiftUI

struct WhisperResult: Equatable {
    let probability: Float
    let text: String
    let token: whisper_token
    var t0: Int64
    var t1: Int64
}

actor WhisperStream {
    private var cancellables: Set<AnyCancellable> = []
    private var context: OpaquePointer?
    private var state: OpaquePointer?
    private var params: whisper_full_params
    private var source: URL
    private var freq: Double
    private var window: Double

    init(model: URL, source: URL, freq: Double, window: Double, lang: UnsafePointer<Int8>) {
        self.freq = freq
        self.window = window
        context = whisper_init_from_file(model.path())
        state = whisper_init_state(context)
        params = whisper_full_default_params(WHISPER_SAMPLING_BEAM_SEARCH)
        params.print_realtime = false
        params.print_progress = false
        params.print_timestamps = false
        params.max_tokens = 0
        params.print_special = false
        params.translate = false
        params.n_threads = 2 // this doesn't matter because we're running on GPU
        params.offset_ms = 0
        params.no_context = true
        params.single_segment = true
        params.audio_ctx = 768 // this makes it faster idk :shrug:
        params.token_timestamps = true
        params.suppress_non_speech_tokens = false
        params.max_len = 1
        params.language = lang
        params.detect_language = false
        self.source = source
    }

    deinit {
        whisper_free(context)
        whisper_free_state(state)
    }

    func start(callback: @escaping (_ results: [WhisperResult], _ runtimeMs: Int) -> Void) {
        let timer = Timer.publish(every: freq, on: .main, in: .common).autoconnect()
        timer
            .scan(0) { totalCalls, _ in
                totalCalls + 1
            }.map { (totalCalls: Int) -> Int in
                Int(Double(totalCalls) * self.freq * 2 * 16000)
            }
            .removeDuplicates()
            .sink { (to: Int) in
                let t0 = Date.now
                let from: Int = max(0,
                                    to - Int(
                                        (self.freq * self.window) * 2 * 16000
                                    )) // get last window only
                print("[\(from)->\(to)]")
                guard from < to else { return }
                if let samples = try? decodeWaveFile(
                    self.source,
                    from: from,
                    to: to,
                    minCount: Int((4 * 16000) + 1)
                ),
                    let context = self.context, let state = self.state
                {
                    self.appendWithState(
                        context: context,
                        state: state,
                        params: self.params,
                        samples: samples
                    )
                    var results = (self.getTranscriptionWithState(context: context, state: state))
                        .filter {
                            !stringStartsWithOpenBracketAndEndsWithCloseBracket($0.text) &&
                                isValidUTF8String($0.text)
                        }
                    let ts0 = Int64(from / (2 * 16))
                    let ts1 = Int64(to / (2 * 16)) - ts0
                    print("[\(ts0)-->\(ts1)] \(results.count)")
                    if results.count > 0 {
                        // TODO: fudge back t0,t1
                        for idx in results.indices {
                            let result: WhisperResult = results[idx]
                            print(result.text, ts0, ts1, result.t0, result.t1)
                            results[idx].t0 = ts0 + result.t0 // add from
                            results[idx].t1 = ts0 + min(ts1, result.t1) // min with to
                        }

//                        // unclear if resetting the state matters
                        whisper_free_state(state)
                        self.state = whisper_init_state(context)
                    }
                    callback(results, Int(Date.now.timeIntervalSince(t0) * 1000))
                }
                if let data = try? Data(contentsOf: self.source), Int(to) > data.count {
                    timer.upstream.connect().cancel()
                }
            }.store(in: &cancellables)
    }

    func appendWithState(
        context: OpaquePointer,
        state: OpaquePointer,
        params: whisper_full_params,
        samples: [Float]
    ) {
        samples.withUnsafeBufferPointer { samples in
            if whisper_full_with_state(
                context,
                state,
                params,
                samples.baseAddress,
                Int32(samples.count)
            ) != 0 {
                print("Failed to run the model")
            } else {
//                    whisper_print_timings(context)
            }
        }
    }

    func getTranscriptionWithState(context: OpaquePointer,
                                   state: OpaquePointer) -> [WhisperResult]
    {
        var results: [WhisperResult] = []
        for i in 0 ..< whisper_full_n_segments_from_state(state) {
            let n_tokens = whisper_full_n_tokens_from_state(state, i)
            let t0 = whisper_full_get_segment_t0_from_state(state, i) * 10 // convert to millis
            let t1 = whisper_full_get_segment_t1_from_state(state, i) * 10 // convert to millis
            for j in 0 ..< n_tokens {
                let token = whisper_full_get_token_id_from_state(state, i, j)
                let probability = whisper_full_get_token_p_from_state(state, i, j)
                let text = String(cString: whisper_full_get_token_text_from_state(
                    context,
                    state,
                    i,
                    j
                ))
                results.append(WhisperResult(
                    probability: probability,
                    text: text,
                    token: token,
                    t0: t0,
                    t1: t1
                ))
            }
        }
        return results
    }
}

@MainActor
class WhisperState: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var canTranscribe = false
    @Published var isRecording = false
    @Published private var base: [WhisperResult] = []
    @Published private var baseMerged: [WhisperResult] = []
    @Published private var tiny: [WhisperResult] = []
    @Published private var tinyMerged: [WhisperResult] = []

    private var whisperContext: WhisperContext?
    private let recorder = Recorder()
    private var recordedFile: URL? = nil
    private var audioPlayer: AVAudioPlayer?
    private var baseStream: WhisperStream?
    private var tinyStream: WhisperStream?
    private var cancellables: Set<AnyCancellable> = []

    private var modelUrl: URL? {
        Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin")
    }

    private var sampleUrl: URL? {
        Bundle.main.url(forResource: "jfk", withExtension: "wav", subdirectory: "samples")
    }

    private enum LoadError: Error {
        case couldNotLocateModel
    }

    override init() {
        super.init()
        do {
            try loadModel()
            canTranscribe = true
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)\n"
        }

        Publishers.Zip(
            $tiny,
            $tiny.dropFirst(1)
        ).sink { prev, next in
            self.tinyMerged = mergeWithHighestProbability(Array(prev + next).sorted(by: {
                $0.t0 < $1.t0
            }))
        }.store(in: &cancellables)
        $base.sink { new in
            var next = self.baseMerged
//            for current in new {
//                if let overlappingPreviousValueIdx = next.firstIndex(where: {
//                    $0.t0 <= current.t0 && $0.t1 >= current.t0 && $0 != current
//                }) {
//                    let overlappingPreviousValue = next[overlappingPreviousValueIdx]
//                    if overlappingPreviousValue.probability < current.probability {
//                        next[overlappingPreviousValueIdx] = current
//                    }
//                } else {
//                    next.append(current)
//                }
//            }
            self.baseMerged = (next + new)//.sorted(by: { $0.t0 < $1.t0 })
//            print("prev: ", prev.map({ $0.text }).joined())
//            print("next: ", next.map({ $0.text }).joined())
//
//            self.baseMerged = prev.reduce([], { partialResult, current in
//                var results = partialResult

//                return results
//            })
//            print("merged: ", self.baseMerged.map({ $0.text }).joined())
        }.store(in: &cancellables)
        Publishers.CombineLatest(
            $baseMerged,
            $tinyMerged
        ).sink { base, _ in
            self.messageLog += base // mergeStreams(stream1: base, stream2: tiny, weight: 3.0)
                .map { "\($0.text) (\($0.t0)-\($0.t1), \(round($0.probability * 100) / 100.0))" }.joined() + "\n\n"
        }.store(in: &cancellables)
        if let source = sampleUrl,
           let baseModel: URL = Bundle.main.url(forResource: "ggml-base.en", withExtension: "bin"),
           let tinyModel: URL = Bundle.main
           .url(forResource: "ggml-tiny.en", withExtension: "bin")
        {
            "en".withCString { lang in
                baseStream = WhisperStream(
                    model: baseModel,
                    source: source,
                    freq: 0.5,
                    window: 8,
                    lang: lang
                )
                Task {
                    guard let stream = self.baseStream else { return }
                    await stream.start { results, _ in
                        self.base = results
                    }
                }
            }
//            tinyStream = WhisperStream(model: tinyModel, source: source, freq: 0.1, window: 10)
//            Task {
//                guard let stream = self.tinyStream else { return }
//                await stream.start { results, _ in
            ////                    self.tiny = results
//                }
//            }
        }
    }

    private func loadModel() throws {
        messageLog += "Loading model...\n"
        if let modelUrl {
            whisperContext = try WhisperContext.createContext(path: modelUrl.path())
            messageLog += "Loaded model \(modelUrl.lastPathComponent)\n"
        } else {
            messageLog += "Could not locate model\n"
        }
    }

    func transcribeSample() async {
        if let sampleUrl {
            await transcribeAudio(sampleUrl)
        } else {
            messageLog += "Could not locate sample\n"
        }
    }

    private func transcribeAudio(_ url: URL) async {
        if !canTranscribe {
            return
        }
        guard let whisperContext else {
            return
        }

        do {
            canTranscribe = false
            messageLog += "Reading wave samples...\n"
            let data = try readAudioSamples(url)
            messageLog += "Transcribing data...\n"
            await whisperContext.fullTranscribe(samples: data)
            let text = await whisperContext.getTranscription()
            messageLog += "Done: \(text)\n"
        } catch {
            print(error.localizedDescription)
            messageLog += "\(error.localizedDescription)\n"
        }

        canTranscribe = true
    }

    private func readAudioSamples(_ url: URL, from: Int = 0, to: Int? = nil) throws -> [Float] {
        stopPlayback()
        try startPlayback(url)
        return try decodeWaveFile(url, from: from, to: to)
    }

    func toggleRecord() async {
        if isRecording {
            await recorder.stopRecording()
            isRecording = false
            if let recordedFile {
                await transcribeAudio(recordedFile)
            }
        } else {
            requestRecordPermission { granted in
                if granted {
                    Task {
                        do {
                            self.stopPlayback()
                            let file = try FileManager.default.url(
                                for: .documentDirectory,
                                in: .userDomainMask,
                                appropriateFor: nil,
                                create: true
                            )
                            .appending(path: "output.wav")
                            try await self.recorder.startRecording(
                                toOutputFile: file,
                                delegate: self
                            )
                            self.isRecording = true
                            self.recordedFile = file
                        } catch {
                            print(error.localizedDescription)
                            self.messageLog += "\(error.localizedDescription)\n"
                            self.isRecording = false
                        }
                    }
                }
            }
        }
    }

    private func requestRecordPermission(response: @escaping (Bool) -> Void) {
        #if os(macOS)
            response(true)
        #else
            AVAudioSession.sharedInstance().requestRecordPermission { granted in
                response(granted)
            }
        #endif
    }

    private func startPlayback(_ url: URL) throws {
        audioPlayer = try AVAudioPlayer(contentsOf: url)
//        audioPlayer?.play()
    }

    private func stopPlayback() {
        audioPlayer?.stop()
        audioPlayer = nil
    }

    // MARK: AVAudioRecorderDelegate

    nonisolated func audioRecorderEncodeErrorDidOccur(_: AVAudioRecorder, error: Error?) {
        if let error {
            Task {
                await handleRecError(error)
            }
        }
    }

    private func handleRecError(_ error: Error) {
        print(error.localizedDescription)
        messageLog += "\(error.localizedDescription)\n"
        isRecording = false
    }

    nonisolated func audioRecorderDidFinishRecording(_: AVAudioRecorder, successfully _: Bool) {
        Task {
            await onDidFinishRecording()
        }
    }

    private func onDidFinishRecording() {
        isRecording = false
    }
}

private func cpuCount() -> Int {
    ProcessInfo.processInfo.processorCount
}

func stringStartsWithOpenBracketAndEndsWithCloseBracket(_ input: String) -> Bool {
    guard input.count >= 2, input.first == "[" || input.first == "<",
          input.last == "]" || input.last == ">"
    else {
        return false
    }
    return true
}

func isValidUTF8String(_ input: String) -> Bool {
    // Create an iterator over the UTF-8 bytes
    var utf8Iterator = input.utf8.makeIterator()

    while let byte = utf8Iterator.next() {
        if byte & 0xC0 == 0x80 {
            // This byte is a continuation byte, skip it
            continue
        }

        let expectedBytes: Int
        if byte & 0x80 == 0x00 {
            expectedBytes = 1
        } else if byte & 0xE0 == 0xC0 {
            expectedBytes = 2
        } else if byte & 0xF0 == 0xE0 {
            expectedBytes = 3
        } else if byte & 0xF8 == 0xF0 {
            expectedBytes = 4
        } else {
            // Invalid UTF-8 byte
            return false
        }

        for _ in 0 ..< expectedBytes - 1 {
            guard let nextByte = utf8Iterator.next(), nextByte & 0xC0 == 0x80 else {
                return false // Not enough continuation bytes
            }
        }
    }

    return true
}

func mergeWithHighestProbability(_ results: [WhisperResult]) -> [WhisperResult] {
    return results.reduce([]) { merged, current -> [WhisperResult] in
        var mergedResults = merged
        if let existingResult = mergedResults
            .first(where: { $0.t0 <= current.t0 && $0.t1 >= current.t0 })
//            .first(where: { $0.t0 <= current.t1 && $0.t1 >= current.t0 })
        {
            if current.probability > existingResult.probability {
                // Replace existing result with higher probability
//                mergedResults.removeAll { $0 == existingResult }
                mergedResults.append(current)
            }
        } else {
            // No overlap, add the current result
            mergedResults.append(current)
        }
        return mergedResults
    }
}

func mergeStreams(stream1: [WhisperResult], stream2: [WhisperResult],
                  weight: Float) -> [WhisperResult]
{
    // Combine the two merged streams with an adjustable weight
    let weightedStream1 = stream1.map { result in
        WhisperResult(probability: result.probability * weight,
                      text: result.text,
                      token: result.token,
                      t0: result.t0,
                      t1: result.t1)
    }

    let weightedStream2 = stream2.map { result in
        WhisperResult(probability: result.probability,
                      text: result.text,
                      token: result.token,
                      t0: result.t0,
                      t1: result.t1)
    }

    // Merge the weighted streams
    return mergeWithHighestProbability((weightedStream1 + weightedStream2)
        .sorted(by: { $0.t0 < $1.t0
        }))
}
