//
//  CPU.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/5/26.
//

import Foundation

// n * m dot m * j = n * j
//func dot(ma:[[Float]], mb: [[Float]]) -> [[Float]] {
//    var ji = 0
//    var result = [[Float]]()
//    for ra in 0..<ma.count { // n
//        var sum: Float = 0.0
//        for ca in 0..<ma[ra].count { // m a00 -> a01
//            let anm = ma[ra][ca]
//            let bmj = mb[ca][ji]
//            sum += anm * bmj
//        }
//    }
//    return result
//}

func transpose(_ a: [[Float]]) -> [[Float]] {
    var result: [[Float]] = Array(repeating: Array(repeating: 0.0, count: a[0].count), count: a.count)
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

func createWeight(rows: Int, columns: Int) -> [[Int]]{
    return [[Int]](repeating: [Int](repeating: 0, count: columns), count: rows)
}