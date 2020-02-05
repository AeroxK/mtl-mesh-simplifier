//
//  main.swift
//  MeshSimplifier
//
//  Created by Alexander König on 25.12.19.
//  Copyright © 2019 Alexander König. All rights reserved.
//

import Foundation
import MetalKit

let device: MTLDevice!

device = MTLCreateSystemDefaultDevice()

let simplifier = Simplifier(device: device)

try simplifier.simplify(filename: "bunny", iterations: 20)
