//
//  ContentView.swift
//  IOSConnect
//
//  Created by Vipaswi Jung Thapa on 5/26/25.
//

import SwiftUI

struct ContentView: View {
    @State private var startSharing = false
    @State private var socket: Int32 = -1

    var body: some View {
        VStack {
            Toggle("Start Sharing", isOn: $startSharing)
                .padding()
                .onChange(of: startSharing) { wasoff, isOn in
                    if isOn {
                        socket = createSocket()
                        if socket < 0 {
                            print("Failed to create socket")
                        } else {
                            startMotionUpdates(socket)
                        }
                    } else {
                        closeSocket(socket)
                        print("Socket closed, sharing ending.")
                    }
                }
        }
    }
}


#Preview {
    ContentView()
}


