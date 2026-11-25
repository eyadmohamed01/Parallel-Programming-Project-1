#include <loss/BinaryCrossEntropy.h>

#include <MathUtils.h>

#include <algorithm>
#include <cmath>

namespace dnn {

std::string BinaryCrossEntropy::getName() const {
    return "binary_cross_entropy";
}

double BinaryCrossEntropy::getCostValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {
    std::vector<double> P(predictedOutput.size(), 0.0);
    std::vector<double> Q(predictedOutput.size(), 0.0);
    std::vector<double> R(predictedOutput.size(), 0.0);

    for(std::size_t i = 0; i < predictedOutput.size(); i++) {
        P[i] = std::log(clampProbability(predictedOutput[i]));
        Q[i] = std::log(1 - clampProbability(predictedOutput[i]));
        R[i] = 1 - actualOutput[i];
    }

    return -((dotProduct(actualOutput, P) + dotProduct(R, Q)) / predictedOutput.size());
}

double BinaryCrossEntropy::getAccuracyValue(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {

    std::vector<double> V(predictedOutput.size(), 0.0);
    std::size_t correct = 0;
    for(std::size_t i = 0; i < predictedOutput.size(); i++) {
        if (predictedOutput[i] > 0.5) {
            V[i] = 1.0;
        }
        if (V[i] == actualOutput[i]) {
            correct++;
        }
    }
    return (static_cast<double>(correct) / predictedOutput.size()); 
}

std::vector<double> BinaryCrossEntropy::getGradient(const std::vector<double>& predictedOutput, const std::vector<double>& actualOutput) const {

    std::vector<double> G(predictedOutput.size(), 0.0);
    for(std::size_t i = 0; i < predictedOutput.size(); i++) {
        G[i] = -(((actualOutput[i] / clampProbability(predictedOutput[i])) - ((1 - actualOutput[i]) / (1 - clampProbability(predictedOutput[i])))) / 
            predictedOutput.size());
    }

    return G;

}

}
