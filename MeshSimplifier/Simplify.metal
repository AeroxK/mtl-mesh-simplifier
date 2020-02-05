//
//  Simplify.metal
//  MeshSimplifier
//
//  Created by Alexander König on 25.12.19.
//  Copyright © 2019 Alexander König. All rights reserved.
//

#include <metal_stdlib>
using namespace metal;

// ### Custom Data Types ###

struct quadric {
    float vars[10];
};

struct arrayElementMetadata {
    int count;
    int offset;
};

struct vertexData {
    bool isDeleted;
    int adjacentFacesCount;
    int offsetInVertexFacesArray;
    int vertexFacesContinueTo;
};

// ### Auxiliary Functions ###

packed_float4 matrix_vector_product(float4x4 matrix, packed_float4 vector) {
    return float4(matrix[0][0] * vector[0] + matrix[1][0] * vector[1] + matrix[2][0] * vector[2] + matrix[3][0] * vector[3],
                  matrix[0][1] * vector[0] + matrix[1][1] * vector[1] + matrix[2][1] * vector[2] + matrix[3][1] * vector[3],
                  matrix[0][2] * vector[0] + matrix[1][2] * vector[1] + matrix[2][2] * vector[2] + matrix[3][2] * vector[3],
                  matrix[0][3] * vector[0] + matrix[1][3] * vector[1] + matrix[2][3] * vector[2] + matrix[3][3] * vector[3]);
}

float scalar_product_homogenous(packed_float4 v1, packed_float4 v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2] + v1[3] * v2[3];
}

float scalar_product(packed_float3 v1, packed_float3 v2) {
    return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

packed_float3 cross_product(packed_float3 v1, packed_float3 v2) {
    return packed_float3(v1[1] * v2[2] - v1[2] * v2[1],
                         v1[2] * v2[0] - v1[0] * v2[2],
                         v1[0] * v2[1] - v1[1] * v2[0]);
}

packed_float3 norm(packed_float3 v) {
    return sqrt(scalar_product(v, v));
}

// TODO: Implement matrix inversion
float4x4 invertMatrix(float4x4 matrix) {
    return matrix;
}

quadric addQuadrics(quadric Q1, quadric Q2) {
    for (int i = 0; i < 10; i++) {
        Q1.vars[i] += Q2.vars[i];
    }
    return Q1;
}

float4x4 getMatrixFromQuadric(quadric Q) {
    return float4x4(Q.vars[0], Q.vars[1], Q.vars[2], Q.vars[3],
                    Q.vars[1], Q.vars[4], Q.vars[5], Q.vars[6],
                    Q.vars[2], Q.vars[5], Q.vars[7], Q.vars[8],
                    Q.vars[3], Q.vars[6], Q.vars[8], Q.vars[9]);
}

float4 getOptimalVertexPosition(quadric q) {
    float4x4 Q = getMatrixFromQuadric(q);
    float4x4 derivateMatrix = float4x4(Q[0][0], Q[1][0], Q[2][0], Q[3][0],
                                       Q[1][0], Q[1][1], Q[2][1], Q[3][1],
                                       Q[2][0], Q[2][1], Q[2][2], Q[3][2],
                                       0., 0., 0., 1.);
    return matrix_vector_product(invertMatrix(derivateMatrix), packed_float4(0., 0., 0., 1.));
}

packed_float3 getMedianVertexPosition(packed_float3 v1, packed_float3 v2) {
    return packed_float3((v1[0]+v2[0])/2, (v1[1]+v2[1])/2, (v1[2]+v2[2])/2);
}

// For each vertex in the face, remove the face from the adjacent face list
void removeFace(int faceIndex,
                device packed_int3* faces,
                device vertexData* vertexData,
                device int* vertexFaces) {
    packed_int3 face = faces[faceIndex];
    for (int i = 0; i < 3; i++) {
        int continuesTo = face[i];
        do {
            struct vertexData currData = vertexData[continuesTo];
            for (int j = currData.offsetInVertexFacesArray + currData.adjacentFacesCount - 1; j >= currData.offsetInVertexFacesArray; j--) {
                if (vertexFaces[j] == faceIndex) {
                    vertexData[continuesTo].adjacentFacesCount = vertexData[continuesTo].adjacentFacesCount - 1;
                    int lastVertexFaceIndex = vertexData[continuesTo].offsetInVertexFacesArray + vertexData[continuesTo].adjacentFacesCount;
                    vertexFaces[j] = vertexFaces[lastVertexFaceIndex];
                    vertexFaces[lastVertexFaceIndex] = -1;
                }
            }
            continuesTo = currData.vertexFacesContinueTo;
        } while (continuesTo != -1);
    }
}

// ### GPU Functions ###

kernel void compute_quadrics_per_face(device packed_float3* vertices,
                    device packed_int3* faces,
                    device quadric* faceQuadrics,
                    uint index [[thread_position_in_grid]]) {

    const packed_int3 currFace = faces[index];
    
    const packed_float3 p = vertices[currFace[0]];
    const packed_float3 v1 = vertices[currFace[1]] - vertices[currFace[0]];
    const packed_float3 v2 = vertices[currFace[2]] - vertices[currFace[0]];
    
    const packed_float3 nVector = cross_product(v1, v2);
    const packed_float3 nVectorNormed = nVector / norm(nVector);
    
    // from the planar equation ax + by + cz + d = 0 we get
    // -(a,b,c) * (x,y,z) = d
    float a = nVectorNormed[0];
    float b = nVectorNormed[1];
    float c = nVectorNormed[2];
    float d = scalar_product(-nVectorNormed, p);
    
    quadric Q;
    Q.vars[0] = a * a; Q.vars[1] = a * b; Q.vars[2] = a * c; Q.vars[3] = a * d;
    Q.vars[4] = b * b; Q.vars[5] = b * c; Q.vars[6] = b * d;
    Q.vars[7] = c * c; Q.vars[8] = c * d;
    Q.vars[9] = d * d;
    
    faceQuadrics[index] = Q;
}

kernel void sum_quadrics_per_vertex(device quadric* faceQuadrics,
                                    device quadric* vertexQuadrics,
                                    device vertexData* vertexData,
                                    device int* vertexFaces,
                                    uint index [[thread_position_in_grid]]) {
    
    // For the current vertex, iterate the corresponding faces using vertexFaces and calculate the sum of their quadrics
    const struct vertexData currVertexData = vertexData[index];
    
    for(
        int i = currVertexData.offsetInVertexFacesArray;
        i < currVertexData.offsetInVertexFacesArray + currVertexData.adjacentFacesCount;
        i++
    ) {
        const int currFace = vertexFaces[i];
        vertexQuadrics[index] = addQuadrics(vertexQuadrics[index], faceQuadrics[currFace]);
    }
}

kernel void collapse_edges(device packed_float3* vertices,
                           device packed_int3* faces,
                           device arrayElementMetadata* independentVerticesMetadata,
                           device int* independentVertices,
                           device vertexData* vertexData,
                           device int* vertexFaces,
                           device quadric* vertexQuadrics,
                           uint index [[thread_position_in_grid]]) {
    
    // For the current independent group of vertices...
    // ...the first vertex in the group order given by memory is the central one
    // ...we need to find the non-central vertex with lowest error
    // ...then we perform an edge collapse on these two
    
    const struct arrayElementMetadata currGroupMetadata = independentVerticesMetadata[index];
    const int centralVertexIndex = independentVertices[currGroupMetadata.offset];
    
    int lowestCostVertexIndex;
    packed_float3 currPos;
    quadric currQ;
    float currError = FLT_MAX;
    
    // Iterate through the current group of independent vertices
    for (int i = currGroupMetadata.offset + 1; i < currGroupMetadata.offset + currGroupMetadata.count; i++) {
        quadric edgeQuadric = addQuadrics(vertexQuadrics[centralVertexIndex], vertexQuadrics[independentVertices[i]]);
        const packed_float3 pos = getMedianVertexPosition(vertices[centralVertexIndex], vertices[independentVertices[i]]);
        const float error = scalar_product_homogenous(packed_float4(pos, 1.), matrix_vector_product(getMatrixFromQuadric(edgeQuadric), packed_float4(pos, 1.)));
        if (error < currError) {
            currError = error;
            currPos = pos;
            currQ = edgeQuadric;
            lowestCostVertexIndex = independentVertices[i];
        }
    }
    
    int continuesTo = centralVertexIndex;
    do {
        for (int i = vertexData[continuesTo].offsetInVertexFacesArray;
             i < vertexData[continuesTo].offsetInVertexFacesArray + vertexData[continuesTo].adjacentFacesCount;
             i++) {
            int currAdjacentFaceIndex = vertexFaces[i];
            if (currAdjacentFaceIndex != -1) {
                for (int currVertex = 0; currVertex < 3; currVertex++) {
                    if (faces[currAdjacentFaceIndex][currVertex] == centralVertexIndex) {
                        faces[currAdjacentFaceIndex][currVertex] = lowestCostVertexIndex;
                    }
                }
            }
        }
        continuesTo = vertexData[continuesTo].vertexFacesContinueTo;
    } while (continuesTo != -1);
    
    vertices[lowestCostVertexIndex] = currPos;
    vertexQuadrics[lowestCostVertexIndex] = currQ;
    
    // Retriangulate area
    vertexData[centralVertexIndex].isDeleted = true;
    
    continuesTo = lowestCostVertexIndex;
    struct vertexData currData = vertexData[continuesTo];
    while(currData.vertexFacesContinueTo != -1) {
        continuesTo = currData.vertexFacesContinueTo;
        currData = vertexData[continuesTo];
    }
    vertexData[continuesTo].vertexFacesContinueTo = centralVertexIndex;
}
