//
//  Predictor.swift
//  CoreMLPerformance
//
//  Created by Vladimir Chernykh on 15.05.2020.
//  Copyright © 2020 Vladimir Chernykh. All rights reserved.
//

import CoreML


class Predictor {

    var model: MLModel

    /// URL of model by filename assuming it was installed in the same bundle as this class
    class func url(forResource fileName: String) -> URL {
        let bundle = Bundle(for: Predictor.self)
        return bundle.url(forResource: fileName, withExtension: "mlmodelc")!
    }

    /**
        Construct a model with explicit path to mlmodelc file
        - parameters:
           - url: the file url of the model
           - throws: an NSError object that describes the problem
    */
    init(contentsOf url: URL) throws {
        self.model = try MLModel(contentsOf: url)
    }

    /**
        Construct a model with explicit path to mlmodelc file and configuration
        - parameters:
           - url: the file url of the model
           - configuration: the desired model configuration
           - throws: an NSError object that describes the problem
    */
    init(contentsOf url: URL, configuration: MLModelConfiguration) throws {
        self.model = try MLModel(contentsOf: url, configuration: configuration)
    }

    /// Construct a model that automatically loads the model from the app's bundle
    convenience init(contentsOf fileName: String) {
        try! self.init(contentsOf: type(of: self).url(forResource: fileName))
    }

    /**
        Construct a model with configuration
        - parameters:
           - fileName: the filename of the model
           - configuration: the desired model configuration
           - throws: an NSError object that describes the problem
    */
    convenience init(contentsOf fileName: String, configuration: MLModelConfiguration) throws {
        try self.init(contentsOf: type(of: self).url(forResource: fileName), configuration: configuration)
    }

    func prediction(input: MLFeatureProvider) throws -> MLFeatureProvider {
        return try model.prediction(from: input)
    }

}
