open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

let vector = Vector<float>.Build
let matrix = Matrix<float>.Build

let historicalReturns = 
    array2D [ 
        [ 0.01;  0.02; -0.01 ]
        [ 0.03;  0.01;  0.00 ]
        [ 0.00;  0.02;  0.01 ]
        [ 0.02; -0.01;  0.03 ]
        [ 0.01;  0.03;  0.02 ]
    ]

let historicalMatrix = matrix.DenseOfArray(historicalReturns)

let meanReturns = historicalMatrix.ColumnSums() / float historicalMatrix.RowCount

let covarianceMatrix =
    let cols = historicalMatrix.ColumnCount
    matrix.Dense(cols, cols, fun i j -> 
        Statistics.Covariance(historicalMatrix.Column(i).ToArray(), historicalMatrix.Column(j).ToArray()))

let portfolioVariance (weights: Vector<float>) =
    weights * covarianceMatrix * weights

let randomWeights numAssets =
    let rawWeights = vector.Random(numAssets)
    rawWeights / rawWeights.Sum()

let findOptimalPortfolio numIterations numAssets =
    let mutable bestWeights = randomWeights numAssets
    let mutable minRisk = portfolioVariance bestWeights
    for _ in 1 .. numIterations do
        let newWeights = randomWeights numAssets
        let newRisk = portfolioVariance newWeights
        if newRisk < minRisk then
            minRisk <- newRisk
            bestWeights <- newWeights
    bestWeights, minRisk

let optimalWeights, minRisk = findOptimalPortfolio 1000 historicalMatrix.ColumnCount

printfn "Оптимальные веса портфеля: %A" optimalWeights
printfn "Минимальный риск: %f" minRisk
