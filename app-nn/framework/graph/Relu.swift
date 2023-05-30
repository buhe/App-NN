//
//  Relu.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/28.
//

import Foundation
class Relu: Layer {
    func bais() -> Float? {
        nil
    }
    
    func weight() -> [[Float]]? {
        nil
    }
    
    var mask: [Int] = []
    func forward(x: [Float]) -> [Float] {
//        print("x \(x)")
        x.enumerated().map { (index, value) in
            if value < 0 {
                mask.append(index)
            }
        }

        let r = x.map { relu($0) }
//        print("relu \(r)")
        return r
    }
    
    func backward(dout: [Float]) -> [Float] {
        let masked = dout.enumerated().map { (index, value) in
            return mask.contains(index) ? 0 : value
        }
        
//        print("mask \(mask)")
//        print("before \(dout)")
//        print("after \(masked)")
        mask = []
        return masked
    }
    
    func relu(_ x: Float) -> Float {
        return max(0, x)
    }
    
}
