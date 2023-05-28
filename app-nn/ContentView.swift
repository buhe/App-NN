//
//  ContentView.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/4/24.
//

import SwiftUI
import CoreData

struct ContentView: View {
    @Environment(\.managedObjectContext) private var viewContext

    @FetchRequest(
        sortDescriptors: [NSSortDescriptor(keyPath: \Item.timestamp, ascending: true)],
        animation: .default)
    private var items: FetchedResults<Item>
    init() {
//        guard let cifar10Data = readCIFAR10DataFromFile(filename: "cifar-10-batches-bin/data_batch_1.bin") else {
//            return
//        }
//        let images = cifar10Data.images
//        let labels = cifar10Data.labels
//
//        print(labels.first!)
//        print(images.first!)
        
        let a: [[Float]] = [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]
        let b: [[Float]] = [[1.0, 4.0], [2.0, 5.0], [3.0, 6.0]]
        let dinput: [[Float]] = [[1.0,1.0], [1.0,1.0]]

        // 调用函数进行矩阵乘法
//        if let result = matrixMultiplication(a, b) {
//            print(result)
//        }
//
//        let w = createWeight(rows: 2, columns: 3)
//        print(w)
        let affine = Affine(w: b)
        let out = affine.forward(x: a)
        let dout = affine.backward(dout: dinput)
        
        print(out)
        print(dout)
        print("dw: \(affine.dw)")
        
        let at = transpose(a)
        
        print(at)
        
        let r: [Float] = [1.0, -1.0]
        let relu = Relu()
        let rout = relu.forward(x: r)
        
        let dr: [Float] = [2.0, 2.0]
        let drelu = relu.backward(dout: dr)
        
        print("rout: \(rout)")
        print("mask: \(relu.mask)")
        print("drelu: \(drelu)")
    }
    var body: some View {
        NavigationView {
            List {
                ForEach(items) { item in
                    NavigationLink {
                        Text("Item at \(item.timestamp!, formatter: itemFormatter)")
                    } label: {
                        Text(item.timestamp!, formatter: itemFormatter)
                    }
                }
                .onDelete(perform: deleteItems)
            }
            .toolbar {
#if os(iOS)
                ToolbarItem(placement: .navigationBarTrailing) {
                    EditButton()
                }
#endif
                ToolbarItem {
                    Button(action: addItem) {
                        Label("Add Item", systemImage: "plus")
                    }
                }
            }
            Text("Select an item")
        }
    }

    private func addItem() {
        withAnimation {
            let newItem = Item(context: viewContext)
            newItem.timestamp = Date()

            do {
                try viewContext.save()
            } catch {
                // Replace this implementation with code to handle the error appropriately.
                // fatalError() causes the application to generate a crash log and terminate. You should not use this function in a shipping application, although it may be useful during development.
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }

    private func deleteItems(offsets: IndexSet) {
        withAnimation {
            offsets.map { items[$0] }.forEach(viewContext.delete)

            do {
                try viewContext.save()
            } catch {
                // Replace this implementation with code to handle the error appropriately.
                // fatalError() causes the application to generate a crash log and terminate. You should not use this function in a shipping application, although it may be useful during development.
                let nsError = error as NSError
                fatalError("Unresolved error \(nsError), \(nsError.userInfo)")
            }
        }
    }
}

private let itemFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.dateStyle = .short
    formatter.timeStyle = .medium
    return formatter
}()

struct ContentView_Previews: PreviewProvider {
    static var previews: some View {
        ContentView().environment(\.managedObjectContext, PersistenceController.preview.container.viewContext)
    }
}
