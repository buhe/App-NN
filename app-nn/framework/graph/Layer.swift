//
//  Layer.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/28.
//

import Foundation

protocol Layer {
    func forward(x: [Float]) -> [Float]
    func backward(dout: [Float]) -> [Float]
    
    func weight() -> [[Float]]?
}
