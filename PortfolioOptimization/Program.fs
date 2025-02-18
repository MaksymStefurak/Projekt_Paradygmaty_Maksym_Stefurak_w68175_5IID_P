open System
open MathNet.Numerics.LinearAlgebra
open MathNet.Numerics.Statistics

// Явно задаем фабрику для работы с матрицами и векторами
let vector = Vector<float>.Build
let matrix = Matrix<float>.Build

// Пример исторических доходностей активов (возвраты активов)
let historicalReturns = 
    array2D [ 
        [ 0.01;  0.02; -0.01 ]
        [ 0.03;  0.01;  0.00 ]
        [ 0.00;  0.02;  0.01 ]
        [ 0.02; -0.01;  0.03 ]
        [ 0.01;  0.03;  0.02 ]
    ]

// Создаем матрицу на основе корректного типа данных
let historicalMatrix = matrix.DenseOfArray(historicalReturns)

// Вычисление средних доходностей активов
let meanReturns = historicalMatrix.ColumnSums() / float historicalMatrix.RowCount

// Вычисление ковариационной матрицы
let covarianceMatrix =
    let cols = historicalMatrix.ColumnCount
    matrix.Dense(cols, cols, fun i j -> 
        Statistics.Covariance(historicalMatrix.Column(i).ToArray(), historicalMatrix.Column(j).ToArray()))

// Функция оценки риска портфеля (дисперсия)
let portfolioVariance (weights: Vector<float>) =
    weights * covarianceMatrix * weights

// Генерация случайных весов портфеля
let randomWeights numAssets =
    let rawWeights = vector.Random(numAssets)
    rawWeights / rawWeights.Sum()

// Поиск оптимального портфеля методом случайного поиска
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

// Запуск оптимизации на 1000 итераций для 3 активов
let optimalWeights, minRisk = findOptimalPortfolio 1000 historicalMatrix.ColumnCount

// Вывод результата
printfn "Оптимальные веса портфеля: %A" optimalWeights
printfn "Минимальный риск: %f" minRisk
