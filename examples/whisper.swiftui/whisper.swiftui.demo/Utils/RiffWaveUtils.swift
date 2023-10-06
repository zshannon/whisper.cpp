import Foundation

func decodeWaveFile(_ url: URL, from: Int = 0, to: Int? = nil, minCount: Int = 0) throws -> [Float] {
    let data = try Data(contentsOf: url)
    var floats: [Float] = stride(from: 44 + from, to: min(data.count, to == nil ? data.count : to! + 44), by: 2).map {
        return data[$0..<$0 + 2].withUnsafeBytes {
            let short = Int16(littleEndian: $0.load(as: Int16.self))
            return max(-1.0, min(Float(short) / 32767.0, 1.0))
        }
    }
    guard minCount > 0 && floats.count < minCount else { return floats }
    let zeroes: [Float] = Array(repeating: 0, count: minCount - floats.count)
    floats.append(contentsOf: zeroes)
    return floats
}
