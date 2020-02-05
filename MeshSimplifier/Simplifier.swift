//
//  Simplifier.swift
//  MeshSimplifier
//
//  Created by Alexander König on 27.12.19.
//  Copyright © 2019 Alexander König. All rights reserved.
//

import Foundation
import MetalKit

struct vertexData {
    var isDeleted: Bool = false
    var adjacentFacesCount: Int32 = 0
    var offsetInVertexFacesArray: Int32 = 0
    var vertexFacesContinueTo: Int32 = -1
}

struct facesPerVertexGeometry {
    var vertexData: [vertexData]!
    var vertexFaces: [[Int32]]
}

struct meshBuffers {
    var vertexBuffer: MTLBuffer
    var facesBuffer: MTLBuffer
    var vertexDataBuffer: MTLBuffer
    var vertexFacesBuffer: MTLBuffer
    var vertexQuadricsBuffer: MTLBuffer
}

class Simplifier {
    let QUADRIC_STORAGE_SIZE: Int!
    let device: MTLDevice!
    let commandQueue: MTLCommandQueue!
    let library: MTLLibrary!
    init(device: MTLDevice) {
        self.device = device
        self.commandQueue = self.device.makeCommandQueue()
        self.library = self.device.makeDefaultLibrary()
        self.QUADRIC_STORAGE_SIZE = 10 * MemoryLayout<Float>.stride
    }
    
    func simplify(filename: String, iterations: Int) throws {
        let initStart = DispatchTime.now()
        
        let mesh: MDLMesh = self.getMeshFromFile(filename: filename)
        
        let submeshes = mesh.submeshes as! [MDLSubmesh]
        assert(mesh.vertexBuffers.count == 1, "Input Mesh Format not matching the constraints")
        assert(submeshes.count == 1, "Input Mesh Format not matching the constraints")
        
        let faceCount = submeshes[0].indexCount / 3
        let vertexCount = mesh.vertexCount
        
        let vBuffer: MDLMeshBuffer = mesh.vertexBuffers[0]
        let iBuffer: MDLMeshBuffer = submeshes[0].indexBuffer
         
        let facesBuffer: MTLBuffer!
        let vertexBuffer: MTLBuffer!
        let faceQuadricsBuffer: MTLBuffer!
        
        facesBuffer = device.makeBuffer(bytes: iBuffer.map().bytes, length: faceCount * 3 * MemoryLayout<Int32>.stride, options: MTLResourceOptions.storageModeShared)
         
        vertexBuffer = device.makeBuffer(bytes: vBuffer.map().bytes, length: vertexCount * 3 * MemoryLayout<Float>.stride, options: MTLResourceOptions.storageModeShared)
            
        faceQuadricsBuffer = device.makeBuffer(length: self.QUADRIC_STORAGE_SIZE * faceCount, options: MTLResourceOptions.storageModeShared)
        
        let faceQuadricsCommandBuffer: MTLCommandBuffer = try self.getFaceQuadrics(
            vertexBuffer: vertexBuffer,
            facesBuffer: facesBuffer,
            faceQuadricsBuffer: faceQuadricsBuffer
        )
        
        let vertexFaceGeometry: facesPerVertexGeometry = self.getFacesPerVertex(
            faces: getArrayFromBuffer(buffer: facesBuffer),
            vertices: getArrayFromBuffer(buffer: vertexBuffer))
        
        // STEP 2: Get Vertex Quadrics
        let vertexDataBuffer: MTLBuffer!
        let vertexFacesBuffer: MTLBuffer!
        let vertexQuadricsBuffer: MTLBuffer!
        
        vertexDataBuffer = device.makeBuffer(bytes: vertexFaceGeometry.vertexData, length: vertexFaceGeometry.vertexData.count * MemoryLayout<vertexData>.stride, options: MTLResourceOptions.storageModeShared)
        
        vertexFacesBuffer = device.makeBuffer(bytes: vertexFaceGeometry.vertexFaces.reduce([], +), length: vertexFaceGeometry.vertexFaces.reduce([], +).count * MemoryLayout<Int32>.stride, options: MTLResourceOptions.storageModeShared)
        
        vertexQuadricsBuffer = device.makeBuffer(length: vertexCount * 10 * MemoryLayout<simd_float1>.stride, options: MTLResourceOptions.storageModeShared)
        
        if (faceQuadricsCommandBuffer.status != MTLCommandBufferStatus.completed) {
            faceQuadricsCommandBuffer.waitUntilCompleted()
        }
        
        let vertexQuadricsCommandBuffer: MTLCommandBuffer = try self.getVertexQuadrics(
            faceQuadricsBuffer: faceQuadricsBuffer,
            vertexQuadricsBuffer: vertexQuadricsBuffer,
            vertexDataBuffer: vertexDataBuffer,
            vertexFacesBuffer: vertexFacesBuffer
        )
    
        // STEP 3: Simplify Mesh
        if (vertexQuadricsCommandBuffer.status != MTLCommandBufferStatus.completed) {
            vertexQuadricsCommandBuffer.waitUntilCompleted()
        }
        
        let initEnd = DispatchTime.now()   // <<<<<<<<<<   end time
        var nanoTime = initEnd.uptimeNanoseconds - initStart.uptimeNanoseconds // <<<<< Difference in nano seconds (UInt64)
        let initTime = Double(nanoTime) / 1_000_000_000 // Technically could overflow for long running tests
        print("Initialization completed in \(initTime) seconds.")
        let iterationStart = DispatchTime.now()
        
        var buffers: meshBuffers = meshBuffers(
            vertexBuffer: vertexBuffer,
            facesBuffer: facesBuffer,
            vertexDataBuffer: vertexDataBuffer,
            vertexFacesBuffer: vertexFacesBuffer,
            vertexQuadricsBuffer: vertexQuadricsBuffer)
        
        for _ in 0..<(iterations) {
            buffers = try iterate(buffers: buffers)
        }
        
        var vertexFaces: [Int32] = getArrayFromBuffer(buffer: buffers.vertexFacesBuffer)
        vertexFaces.removeAll(where: { $0 == -1 })
        vertexFaces = Array(Set(vertexFaces))
        
        var newFaces: [Int32] = [Int32]()
        let faces: Array<Int32> = getArrayFromBuffer(buffer: buffers.facesBuffer)
        for faceIndex in vertexFaces {
            newFaces.append(contentsOf: faces[(Int(faceIndex*3)..<Int(faceIndex*3+3))])
        }
        
        let vertices: [Float] = getArrayFromBuffer(buffer: buffers.vertexBuffer)
        
        let verticesRaw: Data = Data.init(bytes: vertices, count: vertices.count * MemoryLayout<Float>.stride)
        let facesRaw: Data = Data.init(bytes: newFaces, count: newFaces.count * MemoryLayout<Int32>.stride)
        
        let buffer1: MDLMeshBuffer = mesh.allocator.newBuffer(with: verticesRaw, type: MDLMeshBufferType.vertex)
        let buffer2: MDLMeshBuffer = mesh.allocator.newBuffer(with: facesRaw, type: MDLMeshBufferType.index)
        
        let outputMesh: MDLMesh = MDLMesh.init(
         vertexBuffer: buffer1,
         vertexCount: vertices.count / 3,
         descriptor: MDLVertexDescriptor.init(vertexDescriptor: mesh.vertexDescriptor),
         submeshes: [MDLSubmesh.init(indexBuffer: buffer2,
                                     indexCount: newFaces.count,
                                     indexType: MDLIndexBitDepth.uInt32,
                                     geometryType: MDLGeometryType.triangles,
                                     material: MDLMaterial.init())
                    ]
        )
        
        let outputAsset: MDLAsset = MDLAsset.init(bufferAllocator: mesh.allocator)
        outputAsset.add(outputMesh)
        
        let url = Bundle.main.url(forResource: "simplified", withExtension: "obj")!
        try outputAsset.export(to: url)
        
        let iterationEnd = DispatchTime.now()
        nanoTime = iterationEnd.uptimeNanoseconds - iterationStart.uptimeNanoseconds
        let iterationTime = Double(nanoTime) / 1_000_000_000
        print("Completed \(iterations) iterations in \(iterationTime) seconds.")
    }
    
    func iterate(buffers: meshBuffers) throws -> meshBuffers {
        // STEP 1: Get Face Quadrics
        let facesBuffer: MTLBuffer = buffers.facesBuffer
        let vertexBuffer: MTLBuffer = buffers.vertexBuffer
        let vertexDataBuffer: MTLBuffer = buffers.vertexDataBuffer
        let vertexFacesBuffer: MTLBuffer = buffers.vertexFacesBuffer
        let vertexQuadricsBuffer: MTLBuffer = buffers.vertexQuadricsBuffer
        
        let independentVerticesData: [[[Int32]]] = self.getIndependentVertices(
            faces: getArrayFromBuffer(buffer: facesBuffer),
            vertexFaces: getArrayFromBuffer(buffer: vertexFacesBuffer),
            vertexMetadata: getArrayFromBuffer(buffer: vertexDataBuffer)
        )
        let independentVerticesMetadata: [[Int32]] = independentVerticesData[0]
        let independentVertices: [[Int32]] = independentVerticesData[1]
        
        let independentVerticesMetadataBuffer: MTLBuffer!
        let independentVerticesBuffer: MTLBuffer!
        
        independentVerticesMetadataBuffer = device.makeBuffer(bytes: independentVerticesMetadata.reduce([], +), length: independentVerticesMetadata.reduce([], +).count * MemoryLayout<Int32>.stride, options: MTLResourceOptions.storageModeShared)
        
        independentVerticesBuffer = device.makeBuffer(bytes: independentVertices.reduce([], +), length: independentVertices.reduce([], +).count * MemoryLayout<Int32>.stride, options: MTLResourceOptions.storageModeShared)
        
        let edgeCollapseCommandBuffer = try self.collapseEdges(
            vertexBuffer: vertexBuffer,
            facesBuffer: facesBuffer,
            independentVerticesMetadataBuffer: independentVerticesMetadataBuffer,
            independentVerticesBuffer: independentVerticesBuffer,
            vertexDataBuffer: vertexDataBuffer,
            vertexFacesBuffer: vertexFacesBuffer,
            vertexQuadricsBuffer: vertexQuadricsBuffer
        )
        
        if (edgeCollapseCommandBuffer.status != MTLCommandBufferStatus.completed) {
            edgeCollapseCommandBuffer.waitUntilCompleted()
        }
        
        return meshBuffers(
            vertexBuffer: vertexBuffer,
            facesBuffer: facesBuffer,
            vertexDataBuffer: vertexDataBuffer,
            vertexFacesBuffer: vertexFacesBuffer,
            vertexQuadricsBuffer: vertexQuadricsBuffer)
    }
    
    // ### CPU Functions ###
    
    func getArrayFromBuffer<T>(buffer: MTLBuffer) -> Array<T> {
        let elementCount = buffer.length / MemoryLayout<T>.stride
        let ptr = buffer.contents()
        let dataptr = ptr.bindMemory(to: T.self, capacity: elementCount)
        let bufferptr = UnsafeBufferPointer(start: dataptr, count: elementCount)
        return Array(bufferptr)
    }
    
    func getMeshFromFile(filename: String) -> MDLMesh {
        let url = Bundle.main.url(forResource: filename, withExtension: "obj")!
        let asset = MDLAsset.init(url: url)

        guard let object = asset.object(at: 0) as? MDLMesh else {
            fatalError("Failed to get mesh")
        }
        
        return object
    }
    
    func getFacesPerVertex(faces: [Int32], vertices: [Int32]) -> facesPerVertexGeometry {
        
        let dummyData: vertexData = vertexData()
        
        var vertexData: [vertexData] = Array(repeating: dummyData, count: vertices.count / 3)
        var vertexFaces = [[Int32]](repeating: [Int32](), count: vertices.count / 3)
        
        for i in 0..<(faces.count / 3) {
            for j in 0..<(3) {
                let currVertexIndex = faces[i * 3 + j]
                vertexData[Int(currVertexIndex)].adjacentFacesCount += 1
                vertexFaces[Int(currVertexIndex)].append(Int32(i))
            }
        }
        
        var offset: Int32 = 0
        
        for i in 0..<(vertices.count / 3) {
            vertexData[i].offsetInVertexFacesArray = offset
            offset += vertexData[i].adjacentFacesCount
        }
        
        return facesPerVertexGeometry(vertexData: vertexData, vertexFaces: vertexFaces)
    }
    
    func getIndependentVertices(faces: [Int32], vertexFaces: [Int32], vertexMetadata: [vertexData]) -> [[[Int32]]] {
        
        var vertexUsageArray: [Int32] = [Int32](repeating: 0, count: vertexMetadata.count)

        var independentVerticesMetadata = [[Int32]]()
        var independentVertices = [[Int32]]()
        var offset: Int32 = 0
        
        for (centralVertexIndex, currVertexData) in vertexMetadata.enumerated() {
            if (!currVertexData.isDeleted) {
                var areAdjacentVerticesFree: Bool = true
                var currVertices: [Int32] = [Int32]()
                currVertices.append(Int32(centralVertexIndex))

                var continuesTo: Int32 = Int32(centralVertexIndex)
                repeat {
                    let data = vertexMetadata[Int(continuesTo)]
                    for f in vertexFaces[Int(data.offsetInVertexFacesArray)..<Int(data.offsetInVertexFacesArray + data.adjacentFacesCount)] {
                        for v in 0..<(3) {
                            let currVertex = faces[Int(f * 3) + v]
                            if (vertexUsageArray[Int(currVertex)] == 1) {
                                areAdjacentVerticesFree = false;
                            } else {
                                if (!currVertices.contains(currVertex) && !vertexMetadata[Int(currVertex)].isDeleted) { currVertices.append(currVertex) }
                            }
                        }
                    }
                    continuesTo = data.vertexFacesContinueTo
                } while (continuesTo != -1)

                if (areAdjacentVerticesFree && currVertices.count > 1) {
                    independentVerticesMetadata.append([Int32(currVertices.count), offset])
                    independentVertices.append(currVertices)
                    offset += Int32(currVertices.count)
                    
                    for v in currVertices {
                        vertexUsageArray[Int(v)] = 1
                    }
                }
            }
        }
        return [independentVerticesMetadata, independentVertices]
    }
    
    // ### GPU Functions ###
    
    func getFaceQuadrics(vertexBuffer: MTLBuffer, facesBuffer: MTLBuffer, faceQuadricsBuffer: MTLBuffer) throws -> MTLCommandBuffer {
        let quadricsPerFaceFn: MTLFunction!
        let computeEncoder: MTLComputeCommandEncoder!
        let commandBuffer: MTLCommandBuffer!
        
        commandBuffer = self.commandQueue.makeCommandBuffer()
        computeEncoder = commandBuffer.makeComputeCommandEncoder()
        
        quadricsPerFaceFn = self.library.makeFunction(name: "compute_quadrics_per_face")
        let pipelineState = try self.device.makeComputePipelineState(function: quadricsPerFaceFn)
        computeEncoder.setComputePipelineState(pipelineState)
         
        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(facesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(faceQuadricsBuffer, offset: 0, index: 2)
        
        let faceCount = facesBuffer.length / (3 * MemoryLayout<Int32>.size)
        let gridSize = MTLSizeMake(faceCount, 1, 1)
        let threadsPerThreadgroup = MTLSizeMake(min(256, pipelineState.maxTotalThreadsPerThreadgroup, faceCount), 1, 1)
         
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
         
        commandBuffer.commit()
        
        return commandBuffer
    }
    
    func getVertexQuadrics(faceQuadricsBuffer: MTLBuffer,
                           vertexQuadricsBuffer: MTLBuffer,
                           vertexDataBuffer: MTLBuffer,
                           vertexFacesBuffer: MTLBuffer) throws -> MTLCommandBuffer {
        
        let quadricsPerVertexFn: MTLFunction!
        let computeEncoder: MTLComputeCommandEncoder!
        let commandBuffer: MTLCommandBuffer!
        
        commandBuffer = self.commandQueue.makeCommandBuffer()
        computeEncoder = commandBuffer.makeComputeCommandEncoder()
        
        quadricsPerVertexFn = self.library.makeFunction(name: "sum_quadrics_per_vertex")
        let pipelineState = try self.device.makeComputePipelineState(function: quadricsPerVertexFn)
        computeEncoder.setComputePipelineState(pipelineState)

        computeEncoder.setBuffer(faceQuadricsBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(vertexQuadricsBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(vertexDataBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(vertexFacesBuffer, offset: 0, index: 3)
        
        let vertexCount = vertexDataBuffer.length / MemoryLayout<vertexData>.stride
        let gridSize = MTLSizeMake(vertexCount, 1, 1)
        let threadsPerThreadgroup = MTLSizeMake(min(256, pipelineState.maxTotalThreadsPerThreadgroup, vertexCount), 1, 1)
         
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        
        return commandBuffer
    }
    
    func collapseEdges(vertexBuffer: MTLBuffer,
                       facesBuffer: MTLBuffer,
                       independentVerticesMetadataBuffer: MTLBuffer,
                       independentVerticesBuffer: MTLBuffer,
                       vertexDataBuffer: MTLBuffer,
                       vertexFacesBuffer: MTLBuffer,
                       vertexQuadricsBuffer: MTLBuffer) throws -> MTLCommandBuffer {
        
        let collapseEdgesFn: MTLFunction!
        let computeEncoder: MTLComputeCommandEncoder!
        let commandBuffer: MTLCommandBuffer!
        
        commandBuffer = self.commandQueue.makeCommandBuffer()
        computeEncoder = commandBuffer.makeComputeCommandEncoder()
        
        collapseEdgesFn = self.library.makeFunction(name: "collapse_edges")
        let pipelineState = try self.device.makeComputePipelineState(function: collapseEdgesFn)
        computeEncoder.setComputePipelineState(pipelineState)

        computeEncoder.setBuffer(vertexBuffer, offset: 0, index: 0)
        computeEncoder.setBuffer(facesBuffer, offset: 0, index: 1)
        computeEncoder.setBuffer(independentVerticesMetadataBuffer, offset: 0, index: 2)
        computeEncoder.setBuffer(independentVerticesBuffer, offset: 0, index: 3)
        computeEncoder.setBuffer(vertexDataBuffer, offset: 0, index: 4)
        computeEncoder.setBuffer(vertexFacesBuffer, offset: 0, index: 5)
        computeEncoder.setBuffer(vertexQuadricsBuffer, offset: 0, index: 6)
        
        let independentVertexGroupCount = independentVerticesMetadataBuffer.length / (2 * MemoryLayout<Int32>.size)
        let gridSize = MTLSizeMake(independentVertexGroupCount, 1, 1)
        let threadsPerThreadgroup = MTLSizeMake(min(256, pipelineState.maxTotalThreadsPerThreadgroup, independentVertexGroupCount), 1, 1)
        
        computeEncoder.dispatchThreads(gridSize, threadsPerThreadgroup: threadsPerThreadgroup)
        computeEncoder.endEncoding()
        
        commandBuffer.commit()
        
        return commandBuffer
    }
}
