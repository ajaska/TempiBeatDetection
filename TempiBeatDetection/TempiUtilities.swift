//
//  TempiUtilities.swift
//  TempiBeatDetection
//
//  Created by John Scalo on 1/8/16.
//  Copyright © 2016 John Scalo. See accompanying License.txt for terms.

import Foundation
import Accelerate

func tempi_dispatch_delay(delay:Double, closure:()->()) {
    dispatch_after(
        dispatch_time(
            DISPATCH_TIME_NOW,
            Int64(delay * Double(NSEC_PER_SEC))
        ),
        dispatch_get_main_queue(), closure)
}

func tempi_synchronized(lock: AnyObject, closure: () -> ()) {
    objc_sync_enter(lock)
    closure()
    objc_sync_exit(lock)
}

func tempi_is_power_of_2 (n: Int) -> Bool {
    let lg2 = logbf(Float(n))
    return remainderf(Float(n), powf(2.0, lg2)) == 0
}

func tempi_median(a: [Float]) -> Float {
    // I tried to make this an Array extension and failed. See below.
    let sortedArray : [Float] = a.sort( { $0 < $1 } )
    
    if sortedArray.count == 1 {
        return sortedArray[0]
    }
    
    // Technically the median should return the mean of the 2 'center' values, but that's bad for our purposes. Just use the last.
    return sortedArray[sortedArray.count / 2]
}

func tempi_mean(a: [Float]) -> Float {
    // Again, would be better as an Array extension.    
    var mean: Float = 0
    vDSP_meanv(a, 1, &mean, UInt(a.count))
    return mean
}

func tempi_mode(a: [Float], minFrequency: Int) -> Float? {
    // A redefinition of 'mode' suited to our needs. The input values are rounded and nil is returned if a number doesn't occur at least minFrequency times.
    var buckets = [Int : (Int, Float)]()
    for f in a {
        let i = Int(roundf(f))
        if buckets[i] == nil {
            buckets[i] = (0, f)
        }
        buckets[i]!.0 += 1
    }
    
    var modeValue: Float? = nil
    
    // By starting at minFrequency - 1, we require that the mode value occur at least minFrequency times
    var maxFreq = minFrequency - 1
    
    for b in buckets.values {
        if b.0 > maxFreq {
            maxFreq = b.0
            modeValue = b.1
        }
    }
    
    return modeValue
}

func tempi_mode_int(a: [Int]) -> Int {
    var buckets = [Int : Int]()
    for i in a {
        if buckets[i] == nil {
            buckets[i] = 1
        } else {
            buckets[i]! += 1
        }
    }
    
    var modeValue: Int = 0
    
    // By starting at minFrequency - 1, we require that the mode value occur at least minFrequency times
    var maxFreq = 0
    
    for (value, freq) in buckets {
        if freq > maxFreq {
            modeValue = value
            maxFreq = freq
        }
    }
    
    return modeValue
}

func tempi_dump_array(a: [Float]) {
    for f in a {
        print(String(format:"%.03f", f))
    }
}

//extension Array where Element : IntegerArithmeticType {
//    func median() -> Float {
//        let sortedArray : [Float] = a.sort( { $0 < $1 } )
//        var median : Float
//        
//        if sortedArray.count == 1 {
//            return sortedArray[0]
//        }
//        
//        if sortedArray.count % 2 == 0 {
//            let f1 : Float = sortedArray[sortedArray.count / 2 - 1]
//            let f2 : Float = sortedArray[sortedArray.count / 2]
//            median = (f1 + f2) / 2.0
//        } else {
//            median = sortedArray[sortedArray.count / 2]
//        }
//        
//        return median
//    }
//}
