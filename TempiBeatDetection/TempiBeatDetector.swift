//
//  TempiBeatDetector.swift
//  TempiBeatDetection
//
//  Created by John Scalo on 4/27/16.
//  Copyright © 2016 John Scalo. See accompanying License.txt for terms.

import Foundation
import Accelerate
import AVFoundation

typealias TempiBeatDetectionCallback = (
    timeStamp: Double,
    bpm: Float
    ) -> Void

typealias TempiFileAnalysisCompletionCallback = (
    bpms: [(timeStamp: Double, bpm: Float)],
    mean: Float,
    median: Float,
    mode: Float
    ) -> Void

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
    var frequencyBands: Int = 12
    
    var fft: TempiFFT!

    var beatDetectionHandler: TempiBeatDetectionCallback!
    var fileAnalysisCompletionHandler: TempiFileAnalysisCompletionCallback!
    
    private var audioInput: TempiAudioInput!
    private var lastMagnitudes: [Float]!
    
    // For autocorrelation analysis
    private var fluxHistory: [[Float]]! // Holds calculated flux values for each band
    private var fluxHistoryLength: Double = 12.0 // Save the last N seconds of flux values
    private var fluxTimeStamps: [Double]!
    private let correlationValueThreshold: Float = 0.15 // Any correlations less than this are not reported. Higher numbers produce more accuracy but sparser reporting.

    // Audio input
    private var queuedSamples: [Float]!
    private var queuedSamplesPtr: Int = 0
    private var savedTimeStamp: Double!
    
    // Confidence ratings
    private var confidence: Int = 0
    private var lastMeasuredTempo: Float!

    // Timing
    private var analysisInterval: Double = 1.0
    private var lastAnalyzeTime: Double!
    private var startTime: Double!
    private var preRollTime: Double = 3.0
    
    // Time signature detection
    var periodHistory: [[Int]]!
    private let periodHistoryLength = 10
    private var currentTimeSignatureFactor: Float = 4.0
    
    // File-based analysis
    var mediaPath: String!
    var mediaStartTime: Double = 0.0
    var mediaEndTime: Double = 0.0
    var mediaBPMs: [(timeStamp: Double, bpm: Float)]!
    
    // For validation
    var validationSemaphore: dispatch_semaphore_t!
    var tests: [() -> ()]!
    var testSets: [() -> ()]!
    var savePlotData: Bool = false
    var testTotal: Int = 0
    var testCorrect: Int = 0
    var testSetResults: [Float]!
    var testActualTempo: Float = 0
    var currentTestName, currentTestSetName: String!
    var plotFluxValuesDataFile, plotMedianFluxValuesWithTimeStampsDataFile, plotFullBandFluxValuesWithTimeStampsDataFile: UnsafeMutablePointer<FILE>!
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
    
    func stopMicInput() {
        self.audioInput.stopRecording()
    }
    
    private func setupInput() {
        self.queuedSamples = [Float]()
        self.queuedSamplesPtr = 0
    }
#endif

    func startFromFile(url url: NSURL) {
        dispatch_async(dispatch_get_global_queue(0, 0)) { 
            self.reallyStartFromFile(url: url)
        }
    }

    // MARK: - Private stuff

    private func setupCommon() {
        if (self.fft == nil) {
            self.fft = TempiFFT(withSize: self.chunkSize, sampleRate: self.sampleRate)
            self.fft.windowType = TempiFFTWindowType.hanning
        }
        
        self.lastMagnitudes = [Float](count: self.frequencyBands, repeatedValue: 0)
        self.fluxTimeStamps = [Double]()
        self.fluxHistory = [[Float]].init(count: self.frequencyBands, repeatedValue: [Float]())
        self.mediaBPMs = [(timeStamp: Double, bpm: Float)]()
        
        self.periodHistory = [[Int]].init(count: self.frequencyBands, repeatedValue: [Int]())
        
        self.lastMeasuredTempo = nil
        self.confidence = 0
        self.lastAnalyzeTime = nil
        self.startTime = nil
    }

    private func reallyStartFromFile(url url: NSURL) {
        let avAsset: AVURLAsset = AVURLAsset(URL: url)
        
        self.mediaPath = url.absoluteString
        self.setupCommon()
        
        let assetReader: AVAssetReader
        do {
            assetReader = try AVAssetReader(asset: avAsset)
        } catch let e as NSError {
            print("*** AVAssetReader failed with \(e)")
            return
        }
        
        let settings: [String : AnyObject] = [ AVFormatIDKey : Int(kAudioFormatLinearPCM),
                                               AVSampleRateKey : self.sampleRate,
                                               AVLinearPCMBitDepthKey : 32,
                                               AVLinearPCMIsFloatKey : true,
                                               AVNumberOfChannelsKey : 1 ]
        
        let output: AVAssetReaderAudioMixOutput = AVAssetReaderAudioMixOutput.init(audioTracks: avAsset.tracks, audioSettings: settings)
        
        assetReader.addOutput(output)
        
        if !assetReader.startReading() {
            print("assetReader.startReading() failed")
            return
        }
        
        var samplePtr: Int = 0
        
        var queuedFileSamples: [Float] = [Float]()
        
        repeat {
            var status: OSStatus = 0
            guard let nextBuffer = output.copyNextSampleBuffer() else {
                break
            }
            
            let bufferSampleCnt = CMSampleBufferGetNumSamples(nextBuffer)
            
            var bufferList = AudioBufferList(
                mNumberBuffers: 1,
                mBuffers: AudioBuffer(
                    mNumberChannels: 1,
                    mDataByteSize: 4,
                    mData: nil))
            
            var blockBuffer: CMBlockBuffer?
            
            status = CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer(nextBuffer,
                                                                             nil,
                                                                             &bufferList,
                                                                             sizeof(AudioBufferList),
                                                                             nil,
                                                                             nil,
                                                                             kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
                                                                             &blockBuffer)
            
            if status != 0 {
                print("*** CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer failed with error \(status)")
                break
            }
            
            // Move samples from mData into our native [Float] format.
            // (There's probably an better way to do this using UnsafeBufferPointer but I couldn't make it work.)
            for i in 0..<bufferSampleCnt {
                let ptr = UnsafePointer<Float>(bufferList.mBuffers.mData)
                let newPtr = ptr + i
                let sample = unsafeBitCast(newPtr.memory, Float.self)
                queuedFileSamples.append(sample)
            }
            
            // We have a big buffer of audio (whatever CoreAudio decided to give us).
            // Now iterate over the buffer, sending a chunkSize's (e.g. 4096 samples) worth of data to the analyzer and then
            // shifting by hopSize (e.g. 132 samples) after each iteration. If there's not enough data in the buffer (bufferSampleCnt < chunkSize),
            // then add the data to the queue and get the next buffer.
            
            while queuedFileSamples.count >= self.chunkSize {
                let timeStamp: Double = Double(samplePtr) / Double(self.sampleRate)
                
                if self.mediaEndTime > 0.01 {
                    if timeStamp < self.mediaStartTime || timeStamp > self.mediaEndTime {
                        queuedFileSamples.removeFirst(self.hopSize)
                        samplePtr += self.hopSize
                        continue
                    }
                }
                
                let subArray: [Float] = Array(queuedFileSamples[0..<self.chunkSize])
                
                self.analyzeAudioChunk(timeStamp: timeStamp, samples: subArray)
                
                samplePtr += self.hopSize
                queuedFileSamples.removeFirst(self.hopSize)
            }
            
        } while true
        
        if self.fileAnalysisCompletionHandler != nil {
            var bpms = [Float]()
            
            for tuple in self.mediaBPMs {
                bpms.append(tuple.1)
            }
            
            let mean = tempi_mean(bpms)
            let median = tempi_median(bpms)
            let mode = tempi_mode(bpms)
            
            self.fileAnalysisCompletionHandler(bpms: self.mediaBPMs, mean: mean, median: median, mode: mode)
        }
    }
    
    private func analyzeAudioChunk(timeStamp timeStamp: Double, samples: [Float]) {
        let (flux, success) = self.calculateFlux(timeStamp: timeStamp, samples: samples)
        if (!success) {
            return
        }
        
        if self.savePlotData {
            fputs("\(flux)\n", self.plotFluxValuesDataFile)
            fputs("\(timeStamp) \(flux)\n", self.plotMedianFluxValuesWithTimeStampsDataFile)
            var plotStr = ""
            for i in fluxHistory {
                plotStr = plotStr + " \(i.last!)"
            }
            fputs("\(timeStamp)\(plotStr)\n", self.plotFullBandFluxValuesWithTimeStampsDataFile)
        }

        if self.startTime == nil {
            self.startTime = timeStamp
        }
        
        if timeStamp - self.startTime >= self.preRollTime &&
            (self.lastAnalyzeTime == nil || timeStamp - self.lastAnalyzeTime > self.analysisInterval) {
            self.lastAnalyzeTime = timeStamp
            self.analyzeTimer(timeStamp: timeStamp)
        }
        
        self.fluxTimeStamps.append(timeStamp)
        
        // Remove stale flux values.
        while timeStamp - self.fluxTimeStamps.first! > self.fluxHistoryLength {
            self.fluxTimeStamps.removeFirst()
            for i in 0..<self.frequencyBands {
                self.fluxHistory[i].removeFirst()
            }
        }
    }
    
    private func analyzeTimer(timeStamp timeStamp: Double) {
        self.performMultiBandCorrelationAnalysis(timeStamp: timeStamp)
    }
    
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
    
    private func calculateFlux(timeStamp timeStamp: Double, samples: [Float]) -> (flux: Float, success: Bool) {
        self.fft.fftForward(samples)
        
        switch self.frequencyBands {
            case 6:     self.fft.calculateLogarithmicBands(minFrequency: 100, maxFrequency: 5512, bandsPerOctave: 1)
            case 12:    self.fft.calculateLogarithmicBands(minFrequency: 100, maxFrequency: 5512, bandsPerOctave: 2)
            case 30:    self.fft.calculateLogarithmicBands(minFrequency: 100, maxFrequency: 5512, bandsPerOctave: 5)
            default:    assert(false, "Unsupported number of bands.")
        }
        
        // Use the spectral flux+median max algorithm mentioned in https://bmcfee.github.io/papers/icassp2014_beats.pdf .
        // Basically, instead of summing magnitudes across frequency bands we take the log for each band,
        // subtract it from the same band on the last pass, and then find the median of those diffs across
        // frequency bands. This gives a smoother envelope than the summing algorithm.
        
        var diffs: Array = [Float]()
        for i in 0..<self.frequencyBands {
            var mag = self.fft.magnitudeAtBand(i)

            // log requires > 0
            mag = max(mag, 0.00000001)
            
            mag = log10f(mag)
            
            // The 1000.0 here isn't important; just makes the data easier to see in plots, etc.
            let flux: Float = 1000.0 * max(0.0, mag - self.lastMagnitudes[i])
            
            self.lastMagnitudes[i] = mag
            self.fluxHistory[i].append(flux)
            diffs.append(flux)
        }
        
        return (tempi_median(diffs), true)
    }
    
    // MARK: - Autocorrelation analysis
    
    private func performMultiBandCorrelationAnalysis(timeStamp timeStamp: Double) -> Float {
        
        var bpms: [Float] = [Float]()
        var maxCorrValue: Float = 0.0
        
        // Perform the analysis of each band on a separate thread using GCD.
        // (The speedup from parallelism here isn't earth-shattering - in the 5-10% range - 
        // but still seems like the right thing to do...)
        let group = dispatch_group_create()
        
        self.timeSignatureEvaluation(timeStamp: timeStamp)
        
        for i in 0..<self.frequencyBands {
            dispatch_group_async(group, dispatch_get_global_queue(0, 0), {
                let (corr, bpm, period) = self.performSingleCorrelationAnalysis(timeStamp: timeStamp, fluxValues: self.fluxHistory[i])
                tempi_synchronized(self) {
                    if let corr = corr, bpm = bpm {
                        if corr > maxCorrValue {
                            maxCorrValue = corr
                        }
                        bpms.append(bpm)
                    }
                    if let period = period {
                        self.periodHistory[i].append(period)
                    }
                    if self.periodHistory[i].count > self.periodHistoryLength {
                        self.periodHistory[i].removeFirst()
                    }
                }
            })
        }
        
        dispatch_group_wait(group, DISPATCH_TIME_FOREVER)
        
        if maxCorrValue < self.correlationValueThreshold {
            print(String(format: "%.02f: ** low correlation %.02f", timeStamp, maxCorrValue))
            return maxCorrValue
        }
        
        // I think this method makes more sense than taking the median, but there's a slight negative impact on accuracy
        // which is probably related to other issues. Come back to it.
//        var estimatedBPM: Float
//        if let predominantBPM = tempi_custom_mode(bpms, minFrequency: 2) {
//            estimatedBPM = predominantBPM
//        } else {
//            estimatedBPM = tempi_median(bpms)
//        }
        
        let estimatedBPM = tempi_median(bpms)
        
        // Don't allow confidence utilization when doing correlation analysis since the 'confidence'
        // is already built into the correaltion value and we returned early if it weren't high enough.
        self.handleEstimatedBPM(timeStamp: timeStamp, bpm: estimatedBPM, useConfidence: false)
        
        return maxCorrValue
    }
    
    private func performSingleCorrelationAnalysis(timeStamp timeStamp: Double, fluxValues: [Float]) -> (correlationValue: Float?, bpm: Float?, period: Int?) {
        var corr = tempi_autocorr(fluxValues, normalize: true)
        corr = Array(corr[0..<fluxValues.count])
        
        // Get the top 40 correlations
        var maxes = tempi_max_n(corr, n: 40)
        
        // Throw away indices < 50. Those are all 'echoes' of the original signal.
        maxes = maxes.filter({
            // NB: tempi_max_n returns a tuple. The .0 element is the index into the correlation sequence.
            return $0.0 >= 50
        })
        
        if maxes.count == 0 {
            return (nil, nil, nil)
        }
        
        let corrValue: Float = maxes.first!.1
        
        // The index of the first element is the 'lag' (think 'shift') of the signal that correlates the highest. This is our estimated period.
        let period = maxes.first!.0

        if corrValue < self.correlationValueThreshold {
            return (nil, nil, period)
        }
        
        let interval = Float(period) * Float(self.hopSize) / self.sampleRate
        
        // The dominant period might be that of a repeating beat (8th or 4th note) or it might be that of a measure. If it's a measure then we'll
        // need some way to determine whether the song is in a 3-tempo or a 4-tempo before determining the beat interval (and currently we don't have that).
        // We can make a decent guess as to beat vs. measure by comparing the interval to the theoretical shortest possible measure.
        // Why not discard measure-length periods and only rely on periods in the single beat range? Because some rhythms only reveal their period
        // at the scope of a full measure or even two. E.g. the half or full clavé.
        var beatInterval = interval
        let shortestPossibleMeasure = 60.0 / self.maxTempo * 3.0
        let longestPossibleMeasure = 60.0 / self.minTempo * 3.0
        while beatInterval > longestPossibleMeasure {
            beatInterval /= 2.0
        }
        if beatInterval >= shortestPossibleMeasure {
            beatInterval = beatInterval / self.currentTimeSignatureFactor
        }
        
        let mappedInterval = self.mapInterval(Double(beatInterval))
        
        let bpm = 60.0 / Float(mappedInterval)
        
        return (corrValue, bpm, period)
    }
    
    // MARK: -
    
    private func timeSignatureEvaluation(timeStamp timeStamp: Double) {
        var allPeriods = [Int]()
        for i in 0..<self.periodHistory.count {
            if i==11 {
                continue
            }
            for j in self.periodHistory[i] {
                allPeriods.append(j)
            }
        }
        
        if allPeriods.isEmpty {
            return
        }
        
        let shortestPossibleMeasureDuration = 60.0 / self.maxTempo * 3.0
        let longestPossibleMeasureDuration = 60.0 / self.minTempo * 3.0
        let shortestPossibleMeasurePeriod: Int = Int(floorf(shortestPossibleMeasureDuration / (Float(self.hopSize) / self.sampleRate)))
        let longestPossibleMeasurePeriod: Int = Int(floorf(longestPossibleMeasureDuration / (Float(self.hopSize) / self.sampleRate)))
        let shortestPossibleBeatPeriod: Int = Int(longestPossibleMeasureDuration) / 4
        
        var possibleBeatPeriods = [Int]()
        var possibleMeasurePeriods = [Int]()
        
        for i in allPeriods {
            var new_i = i
            if i < shortestPossibleMeasurePeriod {
                while new_i < shortestPossibleBeatPeriod {
                    new_i *= 2
                }
                possibleBeatPeriods.append(new_i)
            } else {
                while new_i > longestPossibleMeasurePeriod {
                    new_i /= 2
                }
                possibleMeasurePeriods.append(new_i)
            }
        }
        
        if possibleBeatPeriods.isEmpty || possibleMeasurePeriods.isEmpty {
            return
        }
        
        let beatMode = tempi_mode_int(possibleBeatPeriods)
        let measureMode = tempi_mode_int(possibleMeasurePeriods)
        
        let factor = Float(measureMode) / Float(beatMode)
        
        if self.tempo(factor, isNearTempo: 3.0, epsilon: 0.06) || self.tempo(factor, isNearTempo: 6.0, epsilon: 0.06) {
            self.currentTimeSignatureFactor = 3.0
        } else {
            self.currentTimeSignatureFactor = 4.0
        }
//        print("Time sig factor: \(factor)")
    }
    
    private func handleEstimatedBPM(timeStamp timeStamp: Double, bpm: Float, useConfidence: Bool = true) {
        var originalBPM = bpm
        var newBPM = bpm
        var multiple: Float = 0.0
        var adjustedConfidence = self.confidence
        
        if !useConfidence {
            adjustedConfidence = 0
        }
        
        if self.lastMeasuredTempo == nil || self.tempo(bpm, isNearTempo: self.lastMeasuredTempo, epsilon: 2.0) {
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
        
        if self.mediaPath != nil {
            self.mediaBPMs.append((timeStamp: timeStamp, bpm: newBPM))
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