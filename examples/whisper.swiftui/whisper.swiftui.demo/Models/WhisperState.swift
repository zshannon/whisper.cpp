import AVFoundation
import Combine
import Foundation
import SwiftUI

actor WhisperStream {
    private var cancellables: Set<AnyCancellable> = []
    private var ctxBig: OpaquePointer?
    private var ctxLittle: OpaquePointer?
    private var paramsBig: whisper_full_params
    private var paramsLittle: whisper_full_params
    private var source: URL

    init(model: URL, source: URL) {
        ctxBig = whisper_init_from_file(model.path())
        ctxLittle = whisper_init_from_file(model.path())
        paramsBig = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        paramsBig.print_realtime = false
        paramsBig.print_progress = false
        paramsBig.print_timestamps = false
        paramsBig.max_tokens = 0
//        params.language = "".cString(using: .utf8)
        paramsBig.print_special = false
        paramsBig.translate = false
        paramsBig.n_threads = 2
        paramsBig.offset_ms = 0
        paramsBig.no_context = true
        paramsBig.single_segment = false

        paramsLittle = whisper_full_default_params(WHISPER_SAMPLING_GREEDY)
        paramsLittle.print_realtime = false
        paramsLittle.print_progress = false
        paramsLittle.print_timestamps = false
        paramsLittle.max_tokens = 0
//        paraLittlelanguage = "".cString(using: .utf8)
        paramsLittle.print_special = false
        paramsLittle.translate = false
        paramsLittle.n_threads = 2
        paramsLittle.offset_ms = 0
        paramsLittle.no_context = true
        paramsLittle.single_segment = false
        self.source = source
//        "en".withCString { en in
//            params.language = en
//        }
    }

    deinit {
        whisper_free(ctxBig)
        whisper_free(ctxLittle)
    }

    func start() {
        let timer = Timer.publish(every: 0.03, on: .main, in: .common).autoconnect()
        timer
            .scan(0) { totalSeconds, _ in
                totalSeconds + 1
            }.map { (totalSeconds: Int) -> Double in
                Double(totalSeconds) * 0.03
            }
            .map { (totalSeconds: Double) -> Double in
//                print("LITTLE: \(totalSeconds)")
                let from = 0
                let to = Int(totalSeconds * 2 * 16000)
//                if let samples = try? decodeWaveFile(self.source, from: from, to: to), let context = self.ctxLittle {
//                    self.append(context: context, params: self.paramsLittle, samples: samples)
//                    let text = self.getTranscription(context: context)
//                    print("little: \(text)")
//                }
                return totalSeconds
            }
            .map { (t: Double) -> Int in Int(t / 2.0) }
            .removeDuplicates()
            .map { (t: Int) -> Int in t * 2 }
            .sink { totalSeconds in
                let t0 = Date.now
                let to = totalSeconds * 2 * 16000
                let from = 0 // max(0, to - (2 * 2 * 16000))
                guard from < to else { return }
                print("BIG: [\(from)-\(to)]")
                if let samples = try? decodeWaveFile(self.source, from: from, to: to), let context = self.ctxBig {
                    self.append(context: context, params: self.paramsBig, samples: samples)
                    let text = self.getTranscription(context: context)
                    // reset context?
//                    context.state = whisper_init_state(context)
                    print("BIG: \(text)")
                }
                if let data = try? Data(contentsOf: self.source), to > data.count {
                    timer.upstream.connect().cancel()
                }
                let t1 = Date.now
                print("big t: \(t1.timeIntervalSince(t0))")
            }.store(in: &cancellables)
    }

    func append(context: OpaquePointer, params: whisper_full_params, samples: [Float]) {
        samples.withUnsafeBufferPointer { samples in
            if whisper_full(context, params, samples.baseAddress, Int32(samples.count)) != 0 {
                print("Failed to run the model")
            } else {
//                    whisper_print_timings(context)
            }
        }
    }

    func getTranscription(context: OpaquePointer) -> String {
        var transcription = ""
        for i in 0 ..< whisper_full_n_segments(context) {
            transcription += String(cString: whisper_full_get_segment_text(context, i))
        }
        return transcription
    }
}

@MainActor
class WhisperState: NSObject, ObservableObject, AVAudioRecorderDelegate {
    @Published var isModelLoaded = false
    @Published var messageLog = ""
    @Published var canTranscribe = false
    @Published var isRecording = false

    private var whisperContext: WhisperContext?
    private let recorder = Recorder()
    private var recordedFile: URL? = nil
    private var audioPlayer: AVAudioPlayer?
    private var stream: WhisperStream?

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
        if let model = modelUrl, let source = sampleUrl {
            stream = WhisperStream(model: model, source: source)
            Task {
                guard let stream = self.stream else { return }
                await stream.start()
            }
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
