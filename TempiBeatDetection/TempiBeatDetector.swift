//
//  TempiBeatDetector.swift
//  TempiBeatDetection
//
//  Created by John Scalo on 4/27/16.
//  Copyright © 2016 John Scalo. See accompanying License.txt for terms.

import Foundation
import Accelerate

typealias TempiBeatDetectionCallback = (
    timeStamp: Double,
    bpm: Float
    ) -> Void

struct TempiPeakInterval {
    var timeStamp: Double
    var magnitude: Float
    var interval: Float
}

class TempiBeatDetector: NSObject {
    
    // All 3 of sampleRate, chunkSize, and hopSize must be changed in conjunction. (Halve one, halve all of them.)
    var sampleRate: Float = 22050
    
    /// The size in samples of the audio buffer that gets analyzed during each pass
    var chunkSize: Int = 2048
    
    /// The size in samples that we skip between passes
    var hopSize: Int = 90
    
    /// Minimum/maximum tempos that the beat detector can detect. Smaller ranges yield greater accuracy.
    var minTempo: Float = 60
    var maxTempo: Float = 220

    /// The number of bands to split the audio signal into. 6, 12, or 30 supported.
    var frequencyBands: Int = 6

    var beatDetectionHandler: TempiBeatDetectionCallback!
    
    private var audioInput: TempiAudioInput!
    private var peakDetector: TempiPeakDetector!
    private var lastMagnitudes: [Float]!
    
    // For peak analysis
    private var peakHistoryLength: Double = 4.0 // The time in seconds that we want intervals for before we do a bucket analysis to predict a tempo
    private var peakHistory: [TempiPeakInterval]! // Stores the last n peaks
    private var bucketCnt: Int = 5 // We'll separate the intervals into this many buckets.
    
    // For autocorrelation analysis
    private var magHistoryLength: Double = 12.0 // Save the last N seconds of computed magnitudes
    private var magMinimumHistoryLengthForAnalysis: Double = 2.0
    private var magHistory: [Float]!
    private var magTimeStamps: [Double]!
    private let correlationValueThreshold: Float = 0.27 // Any correlations less than this are not reported. Higher numbers produce more accuracy but sparser reporting.
    
    // Audio input
    private var queuedSamples: [Float]!
    private var queuedSamplesPtr: Int = 0
    private var savedTimeStamp: Double!
    
    // Peak detection
    private var lastPeakTimeStamp: Double!
    
    // Confidence ratings
    private var confidence: Int = 0
    private var lastMeasuredTempo: Float = 0.0

    private var firstPass: Bool = true
    
    // For validation
    var startTime: Double = 0.0
    var endTime: Double = 0.0
    var savePlotData: Bool = false
    var testTotal: Int = 0
    var testCorrect: Int = 0
    var testSetResults: [Float]!
    var testActualTempo: Float = 0
    var currentTestName, currentTestSetName: String!
    var plotFFTDataFile, plotMarkersFile, plotAudioSamplesFile: UnsafeMutablePointer<FILE>!
    var allow2XResults: Bool = true
    var allowedTempoVariance: Float = 2.0
    
    // MARK: - Public funcs

#if os(iOS)
    func startFromMic() {
        if self.audioInput == nil {
            self.audioInput = TempiAudioInput(audioInputCallback: { (timeStamp, numberOfFrames, samples) in
                self.handleMicAudio(timeStamp: timeStamp, numberOfFrames: numberOfFrames, samples: samples)
                }, sampleRate: self.sampleRate, numberOfChannels: 1)
        }

        self.setupCommon()
        self.setupInput()
        self.audioInput.startRecording()
    }
    
    func stop() {
        self.audioInput.stopRecording()
    }
    
    func setupCommon() {
        self.lastMagnitudes = [Float](count: self.frequencyBands, repeatedValue: 0)
        self.peakHistory = [TempiPeakInterval]()
        self.magHistory = [Float]()
        self.magTimeStamps = [Double]()
        
        self.peakDetector = TempiPeakDetector(peakDetectionCallback: { (timeStamp, magnitude) in
            self.handlePeak(timeStamp: timeStamp, magnitude: magnitude)
            }, sampleRate: self.sampleRate / Float(self.hopSize))
        
        self.peakDetector.coalesceInterval = 0.2
        self.lastPeakTimeStamp = nil
        self.lastMeasuredTempo = 0
        self.confidence = 0
        self.firstPass = true
    }

    private func setupInput() {
        self.queuedSamples = [Float]()
        self.queuedSamplesPtr = 0
    }
#endif
    
    func analyzeAudioChunk(timeStamp timeStamp: Double, samples: [Float]) {
        let (magnitude, success) = self.calculateMagnitude(timeStamp: timeStamp, samples: samples)
        if (!success) {
            return
        }
        
        if self.savePlotData {
            fputs("\(timeStamp) \(magnitude)\n", self.plotFFTDataFile)
        }

        self.peakDetector.addMagnitude(timeStamp: timeStamp, magnitude: magnitude)
        
        self.magHistory.append(magnitude)
        self.magTimeStamps.append(timeStamp)
        
        // Remove stale mags.
        while timeStamp - self.magTimeStamps.first! > self.magHistoryLength {
            self.magTimeStamps.removeFirst()
            self.magHistory.removeFirst()
        }
    }
    
    // MARK: - Private stuff
    
    private func handleMicAudio(timeStamp timeStamp: Double, numberOfFrames:Int, samples:[Float]) {
        
        if (self.queuedSamples.count + numberOfFrames < self.chunkSize) {
            // We're not going to have enough samples for analysis. Queue the samples and save off the timeStamp.
            self.queuedSamples.appendContentsOf(samples)
            if self.savedTimeStamp == nil {
                self.savedTimeStamp = timeStamp
            }
            return
        }
        
        self.queuedSamples.appendContentsOf(samples)
        
        var baseTimeStamp: Double = self.savedTimeStamp != nil ? self.savedTimeStamp : timeStamp
        
        while self.queuedSamples.count >= self.chunkSize {
            let subArray: [Float] = Array(self.queuedSamples[0..<self.chunkSize])
            self.analyzeAudioChunk(timeStamp: baseTimeStamp, samples: subArray)
            self.queuedSamplesPtr += self.hopSize
            self.queuedSamples.removeFirst(self.hopSize)
            baseTimeStamp += Double(self.hopSize)/Double(self.sampleRate)
        }
        
        self.savedTimeStamp = nil
    }

    private func handlePeak(timeStamp timeStamp: Double, magnitude: Float) {
        if (self.lastPeakTimeStamp == nil) {
            self.lastPeakTimeStamp = timeStamp
            return
        }
        
        let interval: Double = timeStamp - self.lastPeakTimeStamp
        
        if self.savePlotData {
            fputs("\(timeStamp) 1\n", self.plotMarkersFile)
        }
        
        let mappedInterval = self.mapInterval(interval)
        let peakInterval = TempiPeakInterval(timeStamp: timeStamp, magnitude: magnitude, interval: Float(mappedInterval))
        self.peakHistory.append(peakInterval)
        
        // If we have enough data, perform autocorrelation. This might generate a bpm reading.
        let shouldPeformCorrelationAnalysis = self.magHistory.count > 2 &&
            (self.magTimeStamps.last! - self.magTimeStamps.first! >= self.magMinimumHistoryLengthForAnalysis)
        if shouldPeformCorrelationAnalysis {
            self.performCorrelationAnalysis(timeStamp: timeStamp)
        }
        
        self.lastPeakTimeStamp = timeStamp
    }
    
    private func calculateMagnitude(timeStamp timeStamp: Double, samples: [Float]) -> (magnitude: Float, success: Bool) {
        let fft: TempiFFT = TempiFFT(withSize: self.chunkSize, sampleRate: self.sampleRate)
        fft.windowType = TempiFFTWindowType.hanning
        fft.fftForward(samples)
        
        switch self.frequencyBands {
            case 6:     fft.calculateLogarithmicBands(minFrequency: 100, maxFrequency: 5512, bandsPerOctave: 1)
            case 12:    fft.calculateLogarithmicBands(minFrequency: 100, maxFrequency: 5512, bandsPerOctave: 2)
            case 30:    fft.calculateLogarithmicBands(minFrequency: 100, maxFrequency: 5512, bandsPerOctave: 5)
            default:    assert(false, "Unsupported number of bands.")
        }
        
        // Use the spectral flux+median max algorithm mentioned in https://bmcfee.github.io/papers/icassp2014_beats.pdf .
        // Basically, instead of summing magnitudes across frequency bands we take the log for each band,
        // subtract it from the same band on the last pass, and then find the median of those diffs across
        // frequency bands. This gives a smoother envelope than the summing algorithm.
        
        var diffs: Array = [Float]()
        for i in 0..<self.frequencyBands {
            var mag = fft.magnitudeAtBand(i)
            
            if mag > 0.0 {
                mag = log10f(mag)
            }
            
            // The 1000.0 here isn't important; just makes the data easier to see in plots, etc.
            let diff: Float = 1000.0 * max(0.0, mag - self.lastMagnitudes[i])
            
            self.lastMagnitudes[i] = mag
            diffs.append(diff)
        }
        
        if self.firstPass {
            // Don't act on the very first pass since there are no diffs to compare.
            self.firstPass = false
            return (0.0, false)
        }

        return (tempi_median(diffs), true)
    }
    
    // MARK: - Autocorrelation analysis
    
    private func performCorrelationAnalysis(timeStamp timeStamp: Double) -> Float {
        var corr = tempi_autocorr(self.magHistory, normalize: true)
        corr = Array(corr[0..<self.magHistory.count])
        
        // Get the top 40 correlations
        var maxes = tempi_max_n(corr, n: 40)
        
        // Throw away indices < 20. Those are all 'echoes' of the original signal.
        maxes = maxes.filter({
            // NB: tempi_max_n returns a tuple. The .0 element is the index into the correlation sequence.
            return $0.0 >= 20
        })
        
        if maxes.count == 0 {
            return 0
        }
        
        let corrValue: Float = maxes.first!.1
        if corrValue < self.correlationValueThreshold {
            print(String(format: "%.02f: ** low correlation %.02f", timeStamp, corrValue))
            return corrValue
        }
        
        // The index of the first element is the 'lag' (think 'shift') of the signal that correlates the highest. This is our estimated period.
        let period = maxes.first!.0
        let measureInterval = Float(period) * Float(self.hopSize) / self.sampleRate
        
        // Now we have to guess whether the song is in 4/4 or something else. Hmm.
        let beatInterval = measureInterval / 4.0
        
        let mappedInterval = self.mapInterval(Double(beatInterval))

        // Divide into 60 to get the bpm.
        let bpm = 60.0 / Float(mappedInterval)
        
        // Don't allow confidence utilization when doing correlation analysis since the 'confidence' 
        // is already built into the correaltion value and we returned early if it weren't high enough.
        self.handleEstimatedBPM(timeStamp: timeStamp, bpm: bpm, useConfidence: false)
        
        return corrValue
    }
    
    // MARK: - Peak analysis

    private func performPeakAnalysis(timeStamp timeStamp: Double) {
        // At this point autocorrelation analysis seems superior to the below alg. in every way, so it's currently unused.
        // Perhaps it will still be useful as a fallback.
        var buckets = [[Float]].init(count: self.bucketCnt, repeatedValue: [Float]())
        
        let minInterval: Float = 60.0/self.maxTempo
        let maxInterval: Float = 60.0/self.minTempo
        let range = maxInterval - minInterval
        
        for i in 0..<self.peakHistory.count {
            let nextInterval = self.peakHistory[i]
            
            // Idx = (interval - minInterval) / range * (bucketCnt - 1)
            var bucketIdx: Int = Int(roundf((nextInterval.interval - minInterval)/range * (Float(self.bucketCnt) - 1.0)))
            bucketIdx = min(bucketIdx, self.bucketCnt - 1)
            buckets[bucketIdx].append(nextInterval.interval)
        }
        
        // Remove stale intervals.
        self.peakHistory = self.peakHistory.filter({
            return timeStamp - $0.timeStamp <= self.peakHistoryLength
        })
        
        // Sort to find which bucket has the most intervals.
        buckets = buckets.sort({ $0.count < $1.count })
        
        // The predominant bucket is the last.
        let predominantIntervals = buckets.last
        
        // Use the median interval for prediction.
        let medianPredominantInterval = tempi_median(predominantIntervals!)
        
        // Divide into 60 to get the bpm.
        let bpm = 60.0 / medianPredominantInterval
        
        self.handleEstimatedBPM(timeStamp: timeStamp, bpm: bpm)
    }
    
    // MARK: -
    
    private func handleEstimatedBPM(timeStamp timeStamp: Double, bpm: Float, useConfidence: Bool = true) {
        var originalBPM = bpm
        var newBPM = bpm
        var multiple: Float = 0.0
        var adjustedConfidence = self.confidence
        
        if !useConfidence {
            adjustedConfidence = 0
        }
        
        if self.lastMeasuredTempo == 0 || self.tempo(bpm, isNearTempo: self.lastMeasuredTempo, epsilon: 2.0) {
            // The tempo remained constant. Bump our confidence up a notch.
            self.confidence = min(10, self.confidence + 1)
        } else if adjustedConfidence > 2 && self.tempo(bpm, isMultipleOf: self.lastMeasuredTempo, multiple: &multiple) {
            // The tempo changed but it's still a multiple of the last. Adapt it by that multiple but don't change confidence.
            originalBPM = bpm
            newBPM = bpm / multiple
        } else {
            // Drop our confidence down a notch
            self.confidence = max(0, self.confidence - 1)
            if useConfidence {
                adjustedConfidence = self.confidence
            }
            if adjustedConfidence > 5 {
                // The tempo changed but our confidence level in the old tempo was high.
                // Don't report this result.
                print(String(format: "%0.2f: IGNORING bpm = %0.2f", timeStamp, newBPM))
                self.lastMeasuredTempo = newBPM
                return
            }
        }
        
        if self.beatDetectionHandler != nil {
            self.beatDetectionHandler(timeStamp: timeStamp, bpm: newBPM)
        }
        
        if originalBPM != newBPM {
            print(String(format:"%0.2f: bpm = %0.2f (adj from %0.2f)", timeStamp, newBPM, originalBPM))
        } else {
            print(String(format:"%0.2f: bpm = %0.2f", timeStamp, newBPM))
        }
        
        self.testTotal += 1
        if self.tempo(newBPM, isNearTempo: self.testActualTempo, epsilon: self.allowedTempoVariance) {
            self.testCorrect += 1
        } else {
            if self.tempo(newBPM, isNearTempo: 2.0 * self.testActualTempo, epsilon: self.allowedTempoVariance) ||
                self.tempo(newBPM, isNearTempo: 0.5 * self.testActualTempo, epsilon: self.allowedTempoVariance) {
                self.testCorrect += 1
            }
        }
        
        self.lastMeasuredTempo = newBPM
    }
    
    private func mapInterval(interval: Double) -> Double {
        var mappedInterval = interval
        let minInterval: Double = 60.0 / Double(self.maxTempo)
        let maxInterval: Double = 60.0 / Double(self.minTempo)
        
        while mappedInterval < minInterval {
            mappedInterval *= 2.0
        }
        while mappedInterval > maxInterval {
            mappedInterval /= 2.0
        }
        return mappedInterval
    }
    
    private func tempo(tempo1: Float, isMultipleOf tempo2: Float, inout multiple: Float) -> Bool
    {
        let multiples: [Float] = [0.5, 0.75, 1.5, 1.33333, 2.0]
        for m in multiples {
            if self.tempo(m * tempo2, isNearTempo: tempo1, epsilon: m * 3.0) {
                multiple = m
                return true
            }
        }
        
        return false
    }

    private func tempo(tempo1: Float, isNearTempo tempo2: Float, epsilon: Float) -> Bool {
        return tempo2 - epsilon < tempo1 && tempo2 + epsilon > tempo1
    }
    
}