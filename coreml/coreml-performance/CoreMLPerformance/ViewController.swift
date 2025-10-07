//
//  ViewController.swift
//  CoreMLPerformance
//
//  Created by Vladimir Chernykh on 14.05.2020.
//  Copyright © 2020 Vladimir Chernykh. All rights reserved.
//

import UIKit
import Vision


class ViewController: UIViewController {

    func url(forResource fileName: String, withExtension ext: String) -> URL {
        let bundle = Bundle(for: ViewController.self)
        return bundle.url(forResource: fileName, withExtension: ext)!
    }

    func testPerformace(modelName: String, device: String, numIter: Int = 100) throws -> Double {

        let config = MLModelConfiguration()
        if device.lowercased() == "ane" {
            config.computeUnits = .all
        } else if device.lowercased() == "gpu" {
            config.computeUnits = .cpuAndGPU
        } else {
            config.computeUnits = .cpuOnly
        }

        let model = try Predictor(contentsOf: modelName, configuration: config)

        let imageFeatureValue = try MLFeatureValue(
            imageAt: url(forResource: "test01", withExtension: "jpg"),
            constraint: model.model.modelDescription.inputDescriptionsByName["input_1"]!.imageConstraint!,
            options: [.cropAndScale: VNImageCropAndScaleOption.centerCrop.rawValue]
        )
        let input = try MLDictionaryFeatureProvider(
            dictionary: ["input_1": imageFeatureValue.imageBufferValue!]
        )

        let startTime = CACurrentMediaTime()
        for _ in 0..<numIter {
            _ = try! model.prediction(input: input)
        }
        let endTime = CACurrentMediaTime()

        return (endTime - startTime) / Double(numIter)
    }

    override func viewDidLoad() {
        super.viewDidLoad()
        for modelName in [
                "densenet121_keras_applications",
            ] {
            for device in ["ANE", "GPU", "CPU"] {
                // warmup
                _ = try! testPerformace(
                    modelName: modelName,
                    device: device,
                    numIter: 20
                )
                // real test
                let latency = try! testPerformace(
                    modelName: modelName,
                    device: device,
                    numIter: 100
                )
                print(modelName)
                print("Latency \(device): \(latency)")
                print("RPS \(device): \(1 / latency)")
                print()
            }
        }
    }

}
