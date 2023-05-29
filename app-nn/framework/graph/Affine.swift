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
    
    var w: [[Float]]
    var b: [Float]
    var dw: [[Float]] = []
    var db: Float = 0.0
    var x: [[Float]] = []
//    var b: [Float]
    init(w: [[Float]],b: [Float]) {
        self.w = w
        self.b = b
    }
    func forward(x: [Float]) -> [Float]{
        self.x = [x]
        var mul = matrixMultiplication(self.x, w)!.first!
//        print("b: \(self.b)")
        for i in 0..<mul.count {
            mul[i] = mul[i] + self.b[i]
        }
        return mul
    }
    
    func backward(dout: [Float]) -> [Float]{
        self.dw = matrixMultiplication(transpose(self.x), [dout])! // 3 2 2 2
        self.db = dout.reduce(0) { $0 + $1 }
        return matrixMultiplication([dout], transpose(self.w))!.first! //   2 3
    }
}
