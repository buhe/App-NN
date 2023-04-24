//
//  app_nnApp.swift
//  app-nn
//
//  Created by 顾艳华 on 2023/4/24.
//

import SwiftUI

@main
struct app_nnApp: App {
    let persistenceController = PersistenceController.shared

    var body: some Scene {
        WindowGroup {
            ContentView()
                .environment(\.managedObjectContext, persistenceController.container.viewContext)
        }
    }
}
