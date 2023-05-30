//
//  SoftmaxWithLoss.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/28.
//

import Foundation

class SoftmaxWithLoss {
    var t: [Float] = []
    var y: [Float] = []
    var loss: Float = 0.0
    func forward(x: [Float], t: [Float]) -> Float {
        self.t = t
        self.y = softmax(x: x)
        self.loss = crossEntropyError(y: self.y, yHat: self.t)
        return self.loss
    }
    
    func backward() -> [Float] {
        var result = [Float]()
        for (index, value) in self.y.enumerated() {
            let diff = value - self.t[index]
            result.append(diff)
        }
//        print("t: \(t)")
//        print("y: \(y)")
//        print("dloss: \(result)")
        return result

    }
}
