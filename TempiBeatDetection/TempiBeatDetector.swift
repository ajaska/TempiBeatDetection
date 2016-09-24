//
//  TempiBeatDetector.swift
//  TempiBeatDetection
//
//  Created by John Scalo on 4/27/16.
//  Copyright © 2016 John Scalo. See accompanying License.txt for terms.

import Foundation
import Accelerate
import AVFoundation

enum TempiBeatDetectionStatus {
    case silence, music
}
typealias TempiBeatDetectionCallback = (
    _ timeStamp: Double,
    _ status: TempiBeatDetectionStatus,
    _ bpm: Float
    ) -> Void

typealias TempiFileAnalysisCompletionCallback = (
    _ bpms: [(timeStamp: Double, bpm: Float)],
    _ mean: Float,
    _ median: Float,
    _ mode: Float
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
    var lastStatus: TempiBeatDetectionStatus!
    
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
    
    // Silence vs. music evaluation
    private var silenceThreshold: Float = -2.0
    private var avgMagnitudeHistory: [Float]!
    private var magHistoryLength: Int {
        get {
            return Int(self.sampleRate / Float(self.hopSize) * 2.0)
        }
    }

    // Timing
    private var analysisInterval: Double = 1.0
    private var lastAnalyzeTime: Double!
    private var startTime: Double!
    private var preRollTime: Double = 3.0
    private var minMeasureDuration: Float {
        get {
            return 60.0 / self.maxTempo * 3.0
        }
    }
    private var minMeasurePeriod: Float {
        get {
            return self.minMeasureDuration * self.sampleRate / Float(self.hopSize)
        }
    }
    private var maxMeasureDuration: Float {
        get {
            return 60.0 / self.minTempo * 4.0
        }
    }
    private var maxMeasurePeriod: Float {
        get {
            return self.maxMeasureDuration * self.sampleRate / Float(self.hopSize)
        }
    }
    private var maxBeatPeriod: Float {
        get {
            return self.maxMeasurePeriod / 3.0
        }
    }
    
    // Time signature detection
    private var currentTimeSignatureFactor: Float!

    // File-based analysis
    var mediaPath: String!
    var mediaStartTime: Double = 0.0
    var mediaEndTime: Double = 0.0
    var mediaBPMs: [(timeStamp: Double, bpm: Float)]!
    
    // For validation
    var validationSemaphore: DispatchSemaphore!
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

    func startFromFile(url: URL) {
        DispatchQueue.global().async {
            self.reallyStartFromFile(url: url)
        }
    }

    // MARK: - Private stuff

    private func setupCommon() {
        if (self.fft == nil) {
            self.fft = TempiFFT(withSize: self.chunkSize, sampleRate: self.sampleRate)
            self.fft.windowType = TempiFFTWindowType.hanning
        }
        
        self.lastMagnitudes = [Float](repeating: 0, count: self.frequencyBands)
        self.fluxTimeStamps = [Double]()
        self.fluxHistory = [[Float]].init(repeating: [Float](), count: self.frequencyBands)
        self.avgMagnitudeHistory = [Float]()
        self.mediaBPMs = [(timeStamp: Double, bpm: Float)]()
        
        self.lastMeasuredTempo = nil
        self.confidence = 0
        self.lastAnalyzeTime = nil
        self.startTime = nil
        self.currentTimeSignatureFactor = nil
    }

    private func reallyStartFromFile(url: URL) {
        let avAsset: AVURLAsset = AVURLAsset(url: url)
        
        self.mediaPath = url.absoluteString
        self.setupCommon()
        
        let assetReader: AVAssetReader
        do {
            assetReader = try AVAssetReader(asset: avAsset)
        } catch let e as NSError {
            print("*** AVAssetReader failed with \(e)")
            return
        }
        
        let settings: [String : AnyObject] = [ AVFormatIDKey : Int(kAudioFormatLinearPCM) as AnyObject,
                                               AVSampleRateKey : self.sampleRate as AnyObject,
                                               AVLinearPCMBitDepthKey : 32 as AnyObject,
                                               AVLinearPCMIsFloatKey : true as AnyObject,
                                               AVNumberOfChannelsKey : 1 as AnyObject ]
        
        let output: AVAssetReaderAudioMixOutput = AVAssetReaderAudioMixOutput.init(audioTracks: avAsset.tracks, audioSettings: settings)
        
        assetReader.add(output)
        
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
                                                                             MemoryLayout<AudioBufferList>.size,
                                                                             nil,
                                                                             nil,
                                                                             kCMSampleBufferFlag_AudioBufferList_Assure16ByteAlignment,
                                                                             &blockBuffer)
            
            if status != 0 {
                print("*** CMSampleBufferGetAudioBufferListWithRetainedBlockBuffer failed with error \(status)")
                break
            }
            
            // Move samples from mData into our native [Float] format.
            let audioBuffer = AudioBuffer(mNumberChannels: bufferList.mBuffers.mNumberChannels,
                                          mDataByteSize: bufferList.mBuffers.mDataByteSize,
                                          mData: bufferList.mBuffers.mData)
            let data = UnsafeRawPointer(audioBuffer.mData)
            for i in 0..<bufferSampleCnt {
                if let sample = data?.load(fromByteOffset: i*4, as: Float.self) {
                    queuedFileSamples.append(sample)
                }
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
            
            self.fileAnalysisCompletionHandler(self.mediaBPMs, mean, median, mode)
        }
    }
    
    private func analyzeAudioChunk(timeStamp: Double, samples: [Float]) {
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
    
    private func analyzeTimer(timeStamp: Double) {
        self.performMultiBandCorrelationAnalysis(timeStamp: timeStamp)
    }
    
    private func handleMicAudio(timeStamp: Double, numberOfFrames:Int, samples:[Float]) {
        
        if (self.queuedSamples.count + numberOfFrames < self.chunkSize) {
            // We're not going to have enough samples for analysis. Queue the samples and save off the timeStamp.
            self.queuedSamples.append(contentsOf: samples)
            if self.savedTimeStamp == nil {
                self.savedTimeStamp = timeStamp
            }
            return
        }
        
        self.queuedSamples.append(contentsOf: samples)
        
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
    
    private func calculateFlux(timeStamp: Double, samples: [Float]) -> (flux: Float, success: Bool) {
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
        
        // Update the avgMagnitudeHistory array for the purposes of music vs. silence eval.
        let avgMag = tempi_mean(self.lastMagnitudes)
        self.avgMagnitudeHistory.append(avgMag)
        let toRemoveCnt = self.avgMagnitudeHistory.count - self.magHistoryLength
        if toRemoveCnt > 0 {
            self.avgMagnitudeHistory.removeFirst(toRemoveCnt)
        }
        
        return (tempi_median(diffs), true)
    }
    
    // MARK: - Autocorrelation analysis
    
    private func performMultiBandCorrelationAnalysis(timeStamp: Double) {
        
        // "Silence" is defined as > 2s with no magnitudes above the silenceThreshold
        let (isSilence, avgMag) = self.isSilence()
        if isSilence {
            if self.lastStatus == nil || self.lastStatus != .silence {
                print("silence mag: \(avgMag)")
                if self.beatDetectionHandler != nil {
                    self.beatDetectionHandler(timeStamp, .silence, 0)
                }
            }
            self.lastStatus = .silence
            return
        } else {
            self.lastStatus = .music
        }

        var bpms: [Float] = [Float]()
        var maxCorrValue: Float = 0.0
        
        // We'll gather time sigs per band and use the most common for the next pass.
        var estimatedTimeSigFactors: [Float] = [Float]()
        
        // Perform the analysis of each band on a separate thread using GCD.
        // (The speedup from parallelism here isn't earth-shattering - in the 5-10% range - 
        // but still seems like the right thing to do...)
        let group = DispatchGroup()
        
        for i in 0..<self.frequencyBands {
            DispatchQueue.global().async(group: group, execute: {
                let (corr, bpm, timeSigFactor) = self.performSingleCorrelationAnalysis(timeStamp: timeStamp, band: i)
                if let corr = corr, let bpm = bpm {
                    tempi_synchronized(self) {
                        if corr > maxCorrValue {
                            maxCorrValue = corr
                        }
                        bpms.append(bpm)
                        if let timeSigFactor = timeSigFactor {
                            estimatedTimeSigFactors.append(timeSigFactor)
                        }
                    }
                }
            })
        }
        
        _ = group.wait(timeout: DispatchTime.distantFuture)
        
        if maxCorrValue < self.correlationValueThreshold {
            print(String(format: "%.02f: ** low correlation %.02f", timeStamp, maxCorrValue))
            return
        }

        if estimatedTimeSigFactors.count > 2 {
            let timeSigMode = tempi_mode(estimatedTimeSigFactors)
            //print("estimated time sig factor: \(timeSigMode); count: \(estimatedTimeSigFactors.count)")
            self.currentTimeSignatureFactor = timeSigMode
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
        
        return
    }
    
    private func performSingleCorrelationAnalysis(timeStamp: Double, band: Int) -> (correlationValue: Float?, bpm: Float?, timeSignatureFactor: Float?) {
        
        let fluxValues = self.fluxHistory[band]
        
        let corr = tempi_autocorr(fluxValues, normalize: true)
        
        assert(corr.count == fluxValues.count, "*** invalid correlation array size")
        
        // Get the top 40 correlations
        var maxes = tempi_max_n(corr, n: 40)
        
        // Throw away indices < 50. Those are all 'echoes' of the original signal.
        maxes = maxes.filter({
            // NB: tempi_max_n returns a tuple. The .0 element is the index into the correlation sequence.
            return $0.0 >= 50
        })
        
        if maxes.isEmpty {
            return (nil, nil, nil)
        }
        
        let corrValue: Float = maxes.first!.1
        
        if corrValue < self.correlationValueThreshold {
            //print("** corr value \(corrValue) was below threshold; band: \(band)")
            return (nil, nil, nil)
        }
        
        // The index of the first element is the 'lag' (think 'shift') of the signal that correlates the highest. This is our estimated period.
        let period = maxes.first!.0
        let interval = Float(period) * Float(self.hopSize) / self.sampleRate
        
        // The dominant period might be that of a repeating beat (8th or 4th note) or it might be that of a measure. If it's a measure then we'll
        // use the estimated time signature from the last pass.
        // We can make a decent guess as to beat vs. measure by comparing the interval to the theoretical shortest possible measure.
        // Why not discard measure-length periods and only rely on periods in the single beat range? Because some rhythms only reveal their period
        // at the scope of a full measure or even two. E.g. the half or full clavé.
        var beatInterval = interval
        let shortestPossibleMeasure = 60.0 / self.maxTempo * 3.0
        if beatInterval >= shortestPossibleMeasure && self.currentTimeSignatureFactor != nil {
            //print("** using time sig factor \(self.currentTimeSignatureFactor)")
            beatInterval = beatInterval / self.currentTimeSignatureFactor
        }
        
        let mappedInterval = self.mapInterval(Double(beatInterval))
        
        let bpm = 60.0 / Float(mappedInterval)
        
        let timeSigFactor = self.estimatedTimeSigFactor(maxes)
        
        //print("timeStamp: \(timeStamp); band: \(band); bpm: \(bpm)")
        
        return (corrValue, bpm, timeSigFactor)
    }
    
    // MARK: -
    
    private func estimatedTimeSigFactor(_ corrTuples: [(Int, Float)]) -> Float! {
        // The basic idea here is: get the dominant measure period (i.e. the longish one), get the dominant beat period (i.e. the shortish one), and divide.
        // If the ratio looks like 3.0 or 6.0 or 12.0, use 3/4; if the ratio looks like 2.0, 4.0, or 8.0, use 4/4.
        
        var estTimeSigFactor: Float!
        
        let possibleBeatPeriods = corrTuples.filter({
            return Float($0.0) <= self.maxBeatPeriod
        })
        
        if !possibleBeatPeriods.isEmpty {
            let dominantBeatPeriod = Float(possibleBeatPeriods.first!.0)
            let possibleMeasurePeriods = corrTuples.filter({
                return Float($0.0) >= self.minMeasurePeriod && Float($0.0) >= dominantBeatPeriod * 1.9 // Instead of 2.0 so we get marginal values
            })
            
            if !possibleMeasurePeriods.isEmpty {
                let dominantMeasurePeriod = Float(possibleMeasurePeriods.first!.0)
                let ratio = dominantMeasurePeriod / dominantBeatPeriod
                //print("measure period: \(dominantMeasurePeriod); beat period: \(dominantBeatPeriod); ratio: \(ratio)")
                
                if self.tempo(ratio, isNearTempo: 3.0, epsilon: 0.1) ||
                    self.tempo(ratio, isNearTempo: 6.0, epsilon: 0.2) ||
                    self.tempo(ratio, isNearTempo: 12.0, epsilon: 0.5) {
                    estTimeSigFactor = 3.0
                } else if self.tempo(ratio, isNearTempo: 2.0, epsilon: 0.1) ||
                    self.tempo(ratio, isNearTempo: 4.0, epsilon: 0.2) ||
                    self.tempo(ratio, isNearTempo: 8.0, epsilon: 0.5) {
                    estTimeSigFactor = 4.0
                }
            }
        }
        
        return estTimeSigFactor
    }

    private func handleEstimatedBPM(timeStamp: Double, bpm: Float, useConfidence: Bool = true) {
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
            self.beatDetectionHandler(timeStamp, .music, newBPM)
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
    
    private func isSilence() -> (Bool, Float) {
        if self.avgMagnitudeHistory.count < self.magHistoryLength {
            return (false, 0.0)
        } else {
            let max = tempi_max(self.avgMagnitudeHistory)
            return (max < self.silenceThreshold, max)
        }
    }
    
    private func mapInterval(_ interval: Double) -> Double {
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
    
    private func tempo(_ tempo1: Float, isMultipleOf tempo2: Float, multiple: inout Float) -> Bool
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

    private func tempo(_ tempo1: Float, isNearTempo tempo2: Float, epsilon: Float) -> Bool {
        return tempo2 - epsilon < tempo1 && tempo2 + epsilon > tempo1
    }
    
}
