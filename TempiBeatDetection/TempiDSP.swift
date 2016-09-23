//
//  TempiDSP.swift
//  TempiBeatDetection
//
//  Created by John Scalo on 5/12/16.
//  Copyright © 2016 John Scalo. All rights reserved.
//

import Foundation
import Accelerate

func tempi_autocorr(_ a: [Float], normalize: Bool) -> [Float] {
    // vDSP_conv is horribly underdocumented so I'll spell it out for posterity:
    // vDSP_conv returns a correlation sequence for the provided arrays. The index to the output sequence corresponds to the correlation 'lag' (think 'shift') and the value is the non-normalized correlation.
    // When the arrays are the same, vDSP_conv is performing an autocorrelation and returns an autocorrelation sequence.
    // In the autocorrelation scenario, the non-normalized correlation values are simply: sum(a[i]*a[i+p]) for all lags p.
    // Since the signal will always correlate perfectly with itself, the value at index 0 will always be the highest in the sequence.
    // Unfortunately vDSP_conv requires the arrays to be padded with 0s on either side, presumably to make SIMD operations possible.
    // (This assumes strides are 1. For -1, vDSP_conv performs a convolution.)
    let filterLen: UInt = UInt(a.count)
    let resultLen: UInt = filterLen * 2 - 1
    
    // From https://developer.apple.com/library/mac/samplecode/vDSPExamples/Listings/DemonstrateConvolution_c.html :
    // “The signal length is padded a bit. This length is not actually passed to the vDSP_conv routine; it is the number of elements
    // that the signal array must contain. The SignalLength defined below is used to allocate space, and it is the filter length
    // rounded up to a multiple of four elements and added to the result length. The extra elements give the vDSP_conv routine
    // leeway to perform vector-load instructions, which load multiple elements even if they are not all used. If the caller did not
    // guarantee that memory beyond the values used in the signal array were accessible, a memory access violation might result.”
    let signalLen: UInt = ((filterLen + 3) & 0xFFFFFFFC) + resultLen
    
    let padding1 = [Float].init(repeating: 0.0, count: a.count - 1)
    let padding2 = [Float].init(repeating: 0.0, count: (Int(signalLen) - padding1.count - a.count))
    let signal = padding1 + a + padding2
    
    var result = [Float].init(repeating: 0.0, count: Int(resultLen))
    
    vDSP_conv(signal, 1, a, 1, &result, 1, resultLen, filterLen)
    
    // Remove the first n-1 values which are just mirrored from the end. This way [0] always shows the 1.0 autocorrelation.
    result.removeFirst(Int(filterLen) - 1)
    
    if normalize {
        tempi_normalize(&result, a)
    }
    
    return result
}

func tempi_normalize(_ a: inout [Float], _ b: [Float]) {
    // Normalize the values in a using the sum of the squares of values in b
    var sqrs: [Float] = [Float].init(repeating: 0, count: b.count)
    var sum: Float = 0
    var newResult: [Float] = [Float].init(repeating: 0, count: a.count)
    vDSP_vsq(b, 1, &sqrs, 1, UInt(b.count))
    vDSP_sve(sqrs, 1, &sum, UInt(sqrs.count))
    vDSP_vsdiv(a, 1, &sum, &newResult, 1, UInt(a.count))
    a = newResult
}

func tempi_max(_ a: [Float]) -> Float {
    var max: Float = 0.0
    
    vDSP_maxv(a, 1, &max, UInt(a.count))
    return max
}

func tempi_max_n(_ a: [Float], n: Int) -> [(Int, Float)] {
    // Return the indices and values of the greatest n values in a
    var x = a
    var result = [(Int, Float)]()
    
    for _ in 0..<n {
        var max: Float = 0.0
        var idx: UInt = 0
        vDSP_maxvi(x, 1, &max, &idx, UInt(x.count))
        result.append((Int(idx), max))
        x[Int(idx)] = -FLT_MAX
    }
    
    return result
}

func tempi_smooth(_ a: [Float], w: Int) -> [Float] {
    var newA: [Float] = [Float]()
    
    for i in 0..<a.count {
        let realW = min(w, a.count - i)
        var avg: Float = 0.0
        let subArray: [Float] = Array(a[i..<i+realW])
        vDSP_meanv(subArray, 1, &avg, UInt(realW))
        newA.append(avg)
    }
    
    return newA
}
