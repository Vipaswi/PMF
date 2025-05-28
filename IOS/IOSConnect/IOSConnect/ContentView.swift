//
//  ContentView.swift
//  IOSConnect
//
//  Created by Vipaswi Jung Thapa on 5/26/25.
//

import SwiftUI

struct ContentView: View {
    var body: some View {
        VStack {
            Button(action: {
                // Your action here
                
            }) {
                Text("Start Sharing")
                    .frame(width: 100, height: 150)
                    .background(Color.red)
                    .foregroundColor(.white)
                    .cornerRadius(10)
            }
        }
        .padding()
    }
}

#Preview {
    ContentView()
}


