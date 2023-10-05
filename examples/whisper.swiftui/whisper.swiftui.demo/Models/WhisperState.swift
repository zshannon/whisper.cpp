import AVFoundation
import Combine
import Foundation
import SwiftUI

actor WhisperStream {
    private var cancellables: Set<AnyCancellable> = []
    private var context: OpaquePointer?
    
    init(model: URL, source: URL) {
        self.context = whisper_init_from_file(model.path())
    }
    
    deinit {
        whisper_free(context)
    }
    
    func start() {
        Timer.publish(every: 0.03, on: .main, in: .common)
            .autoconnect()
            .scan(0) { totalSeconds, _ in
                totalSeconds + 1
            }.map { (totalSeconds: Int) -> Double in
                Double(totalSeconds) * 0.03
            }
            .map { (totalSeconds: Double) -> Double in
                print("LITTLE: \(totalSeconds)")
                return totalSeconds
            }
            .map { (t: Double) -> Int in Int(t / 2.0) }
            .removeDuplicates()
            .map { (t: Int) -> Int in t * 2 }
            .sink { totalSeconds in
                print("BIG: \(totalSeconds)")
            }.store(in: &cancellables)
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
            let data = try readAudioSamples(url, from: 0, to: 2 * 2 * 16000)
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
