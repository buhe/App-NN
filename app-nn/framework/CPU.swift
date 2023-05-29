//
//  CPU.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/26.
//

import Foundation


func transpose(_ a: [[Float]]) -> [[Float]] {
    var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: a.count), count: a[0].count)
    for i in 0..<a.count {
        for j in 0..<a[i].count {
            result[j][i] = a[i][j]
        }
    }
    return result
}

func matrixMultiplication(_ a: [[Float]], _ b: [[Float]]) -> [[Float]]? {
    
    let rowsInA = a.count
    let columnsInA = a[0].count
    
    let rowsInB = b.count
    let columnsInB = b[0].count
    
    if columnsInA != rowsInB {
        return nil // 不符合矩阵乘法规则
    }
    
    var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: columnsInB), count: rowsInA)
    
    for i in 0 ..< rowsInA {
        for j in 0 ..< columnsInB {
            for k in 0 ..< columnsInA {
                result[i][j] += a[i][k] * b[k][j]
            }
        }
    }
    
    return result
}

func generateZeroArray(size: Int) -> [Float] {
    [Float](repeating: 0, count: size)
}

func generate2DGaussianArray(numRows: Int, numCols: Int, stdRow: Float, stdCol: Float) -> [[Float]] {
    let meanRow = Array(repeating: Float(numRows/2), count: numRows)
    let meanCol = Array(repeating: Float(numCols/2), count: numCols)
    var array2D: [[Float]] = Array(repeating: Array(repeating: 0.0, count: numCols), count: numRows)

    for i in 0..<numRows {
        for j in 0..<numCols {
            let x = Float(j) - meanCol[j]
            let y = Float(i) - meanRow[i]
            let sigmaX = stdCol
            let sigmaY = stdRow
            let exponent = -(powf(x, 2)/(2*powf(sigmaX, 2)) + powf(y, 2)/(2*powf(sigmaY, 2)))
            let value = powf(Float(M_E), exponent)/(2*Float.pi*sigmaX*sigmaY)
            array2D[i][j] = value
        }
    }

    return array2D
}




func softmax(x: [Float]) -> [Float]{
    let expArr = x.map { exp($0) }
    let sumExp = expArr.reduce(0, +)
    return expArr.map { $0 / sumExp }
}

func crossEntropyError(y: [Float], yHat: [Float]) -> Float {
    var total: Float = 0.0
    for i in 0..<y.count {
        let yi = y[i] + 1e-7
        total += yi * log(yHat[i] + 1e-7)
    }
    return -total

}
