//
//  Relu.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/28.
//

import Foundation
class Relu: Layer {
    func weight() -> [[Float]]? {
        nil
    }
    
    var mask: [Int] = []
    func forward(x: [Float]) -> [Float] {
        x.enumerated().map { (index, value) in
            if value < 0 {
                mask.append(index)
            }
        }

        return x.map { relu($0) }
    }
    
    func backward(dout: [Float]) -> [Float] {
        return dout.enumerated().map { (index, value) in
            return mask.contains(index) ? 0 : value
        }
    }
    
    func relu(_ x: Float) -> Float {
        return max(0, x)
    }
    
}
