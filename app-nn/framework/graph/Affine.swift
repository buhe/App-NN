//
//  Affine.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/28.
//

import Foundation

class Affine: Layer {
    func bais() -> Float? {
        db
    }
    
    func weight() -> [[Float]]? {
        dw
    }
    var net: Network
    var w: String
    var b: String
    var dw: [[Float]] = []
    var db: Float = 0.0
    var x: [[Float]] = []
//    var b: [Float]
    init(net: Network, w: String,b: String) {
        self.w = w
        self.b = b
        self.net = net
    }
    func forward(x: [Float]) -> [Float]{
        self.x = [x]
        var mul = matrixMultiplication(self.x, net.params[w]!)!.first! // 1 2 . 2 3 = 1 3
//        print("x: \(x)")
//        print("w: \(net.params[w]!)")
//        print("b: \(net.paramsB[b]!)")
        for i in 0..<mul.count {
            mul[i] = mul[i] + net.paramsB[b]![i]
        }
//        print("affine \(w) \(mul)")
        return mul
    }
    
    func backward(dout: [Float]) -> [Float]{
        self.dw = matrixMultiplication(transpose(self.x), [dout])! // 2 1 . 1 3
        self.db = dout.reduce(0) { $0 + $1 }
        return matrixMultiplication([dout], transpose(net.params[w]!))!.first! //   1 3 . 3 2 = 1 2
    }
}
