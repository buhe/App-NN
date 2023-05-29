//
//  Network.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/28.
//
import SwiftUI
import Foundation

struct Network {
    @State var params: [String: [[Float]]] = [:]
    var layers: [Layer] = []
    var lossLayer = SoftmaxWithLoss()
    init(input_size: Int, hidden_size: Int, output_size: Int) {
        let stdRow: Float = 1.0
        let stdCol: Float = 1.0
        
        params["w1"] = generate2DGaussianArray(numRows: input_size, numCols: hidden_size, stdRow: stdRow, stdCol: stdCol)
        params["w2"] = generate2DGaussianArray(numRows: hidden_size, numCols: output_size, stdRow: stdRow, stdCol: stdCol)

        
        
        layers.append(Affine(w: params["w1"]!))
        layers.append(Relu())
        layers.append(Affine(w: params["w2"]!))
    }
    
    func predict(x: [Float]) -> [Float] {
        var result = x
        for l in layers {
            result = l.forward(x: result)
        }
        return result
    }
    
    func loss(x: [Float], t: [Float]) -> Float {
        let y = self.predict(x: x)
        return lossLayer.forward(x: y, t: t)
    }
    
    func gradient(x: [Float], t: [Float]) -> [String: [[Float]]] {
        let loss = self.loss(x: x, t: t)
        print("loss: \(loss)")
        
        var dout = lossLayer.backward()
        
        let rlayers = Array(layers.reversed())
        for l in rlayers {
            dout = l.backward(dout: dout)
        }
        var grads = [String: [[Float]]]()
        grads["w1"] = layers[0].weight()!
        grads["w2"] = layers[2].weight()!
        return grads
    }
    
    func update(grad: [String: [[Float]]]) {
        updateW(key: "w1", grad: grad)
        updateW(key: "w2", grad: grad)
        
        //        net.params["w1"] -= learning_rate * grad["w1"]
        //        net.params["w2"] -= learning_rate * grad["w2"]
    }
    
    private func updateW(key: String, grad: [String: [[Float]]]){
        let learning_rate: Float = 0.1
        let w = params[key]!
        var newW: [[Float]] = Array(repeating: Array(repeating: 0, count: w[0].count), count: w.count)
        let wg = grad[key]!
        for r in 0..<w.count {
            for c in 0..<w[0].count {
                let e = w[r][c]
                let g = wg[r][c]
                newW[r][c] = e - learning_rate * g
            }
        }
        params[key] = newW
    }
}
