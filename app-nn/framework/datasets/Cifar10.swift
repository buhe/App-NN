//
//  Cifar10.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/24.
//

import Foundation
func toOneHot(labels: [Int]) -> [[Int]] {
    var oneHots = [[Int]]()
    for i in 0..<labels.count {
        var oneHot = [Int](repeating: 0, count: 10)
        oneHot[labels[i]] = 1
        oneHots.append(oneHot)
    }
    return oneHots
    
}
func readCIFAR10DataFromFile(filename: String) -> (images: [[Float]], labels: [[Int]])? {
    guard let fileurl = Bundle.main.url(forResource: filename, withExtension: nil) else {
        return nil
    }
    let data = try! Data(contentsOf: fileurl)
    
    var images = [[Float]]()
    var labels = [Int]()
    for i in 0..<10000 {
        let offset = i * 3073
        let label = Int(data[offset])
        labels.append(label)
        var image = [Float]()
        for j in 1...3072 {
            image.append(Float(data[offset+j]) / 255.0)
        }
        images.append(image)
    }
    return (images, toOneHot(labels: labels))
}


