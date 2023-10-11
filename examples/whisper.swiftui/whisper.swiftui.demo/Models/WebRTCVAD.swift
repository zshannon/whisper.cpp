//
//  WebRTCVAD.swift
//  whisper.swiftui
//
//  Created by Zane Shannon on 10/10/23.
//

import Foundation
import WebRTC

actor WebRTCVAD {
    
//    private let vad = RTCVA
    
    init() {
        let sr = 16000
        let frame_size = 10 // ms
        let frame: [Float] = .init(repeating: 0, count: 2 * sr * frame_size)
//        print("is speech?: ", vad)
    }
    
}
