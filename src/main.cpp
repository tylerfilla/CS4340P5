/*
 * Tyler Filla
 * CS 4340 - Project 5
 * December 13, 2018
 */

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>
#include <utility>

/**
 * A real-valued 2D vector.
 */
struct Vec2
{
    /** The first dimension. */
    double x;

    /** The second dimension. */
    double y;

    /**
     * Sum (vector-vector).
     */
    inline Vec2 operator+(const Vec2& rhs) const
    { return {x + rhs.x, y + rhs.y}; }

    inline Vec2& operator+=(const Vec2& rhs)
    { return *this = *this + rhs; }

    /**
     * Difference (vector-vector).
     */
    inline Vec2 operator-(const Vec2& rhs) const
    { return {x - rhs.x, y - rhs.y}; }

    inline Vec2& operator-=(const Vec2& rhs)
    { return *this = *this - rhs; }

    /**
     * Product (vector-scalar).
     */
    inline Vec2 operator*(double rhs) const
    { return {x * rhs, y * rhs}; }

    inline Vec2& operator*=(double rhs)
    { return *this = *this * rhs; }

    /**
     * Quotient (vector-scalar).
     */
    inline Vec2 operator/(double rhs) const
    { return {x / rhs, y / rhs}; }

    inline Vec2& operator/=(double rhs)
    { return *this = *this / rhs; }

    /**
     * Product (dot, vector-vector).
     */
    inline double dot(const Vec2& rhs) const
    { return x * rhs.x + y * rhs.y; }

    /**
     * Magnitude.
     */
    inline double mag() const
    { return std::sqrt(dot(*this)); }
};

/**
 * Print a vector to an output stream.
 *
 * @param out The output stream
 * @param vec The vector
 * @return The output stream
 */
static std::ostream& operator<<(std::ostream& out, const Vec2& vec)
{
    return out << "(" << vec.x << ", " << vec.y << ")";
}

/**
 * An iterable range.
 */
template<class InputIt>
struct range
{
    InputIt m_begin;
    InputIt m_end;

    range(InputIt p_begin, InputIt p_end)
            : m_begin {p_begin}
            , m_end {p_end}
    {
    }

    InputIt begin()
    { return m_begin; }

    InputIt end()
    { return m_end; }
};

/**
 * Do linear regression over a set of points with optional L2-regularization.
 *
 * @param begin An iterator to the first point
 * @param end An iterator one past the end of the last point
 * @param lambda The regularization constraint (optional, defaults to zero)
 * @return A pair with the final weight vector and the final MSE
 */
template<class InputIt>
static std::pair<Vec2, double> linreg(InputIt begin, InputIt end, double lambda = 0)
{
    // GD iteration limit
    const int il = 1000000;

    // GD learning rate
    const double lr = 0.005;

    // GD termination threshold (magnitude)
    const double th = 0.001;

    // The sample count
    auto N = std::distance(begin, end);

    std::cout << "Performing linear regression on " << N << " points\n";
    std::cout << " * Optimizer: batch gradient descent\n";
    std::cout << " * Iteration limit: " << il << "\n";
    std::cout << " * Learning rate: " << lr << "\n";
    std::cout << " * Target gradient magnitude: " << th << "\n";

    // The weight vector
    Vec2 weight {};

    std::cout << "Initial weight vector: " << weight << "\n";

    // Mean-squared error
    double mse;

    // Total number of iterations taken
    int iters = 0;

    // Gradient descent
    for (int t = 0; t < il; ++t)
    {
        iters = t + 1;
//      std::cout << "Iteration " << (iters = t + 1) << "\n";
//      std::cout << " -> Current weight vector: " << weight << "\n";

        // The gradient vector
        Vec2 grad {};

        mse = 0;

        // Go over all points
        for (auto&& pt : range {begin, end})
        {
            // Make feature vector for point
            Vec2 feature {1, pt.x};

            // Update MSE as-is
            mse += std::pow(pt.y - weight.dot(feature), 2);

            // Update gradient vector
            grad += feature * (pt.y - weight.dot(feature)) * 2;
        }

        // Finalize MSE
        mse /= N;

//      std::cout << " -> Computed MSE: " << mse << "\n";

        // Finalize MSE component of gradient vector
        // We still need to do regularization with the penality term
        grad *= (-1.0 / N);

        // Add in L2 penalty term
        grad += weight * 2 * lambda;

//      std::cout << " -> Computed gradient: " << grad << "\n";

        // If gradient is under threshold magnitude, we're done
        if (grad.dot(grad) <= std::pow(th, 2))
        {
            std::cout << " -> !!! STOP: GRADIENT MINIMIZED (mag. " << grad.mag() << " <= " << th << ")\n";
            break;
        }

        if (iters == il)
        {
            std::cout << " -> !!! STOP: DID NOT CONVERGE\n";
            break;
        }

        // Update weight vector by gradient
        weight -= grad * lr;

//      std::cout << " -> New weight vector: " << weight << "\n";
    }

    std::cout << "Concluding linear regression after " << iters << " iteration(s)\n";
    std::cout << "Final weight vector: " << weight << "\n";
    std::cout << "Final MSE: " << mse << "\n";

    return {weight, mse};
}

/**
 * Do validation over a set of points with a given model.
 *
 * @param begin An iterator to the first point
 * @param end An iterator one past the end of the last point
 * @param model The model under consideration
 * @return A pair with the final weight vector and the final MSE
 */
template<class InputIt>
static double validate(InputIt begin, InputIt end, const Vec2& model)
{
    // Count points
    const auto N = std::distance(begin, end);

    // Mean-squared error
    double mse = 0;

    // Go over all points
    for (auto&& pt : range {begin, end})
    {
        // Feature vector
        Vec2 feature {1, pt.x};

        // Compute MSE term
        mse = std::pow(pt.y - model.dot(feature), 2);
    }

    // Compute MSE
    mse /= N;

    return mse;
}

int main()
{
    std::cout << "Tyler Filla\n";
    std::cout << "CS 4340 - Project 5\n";
    std::cout << "December 13, 2018\n\n";

    // The smallest representable double
    auto epsilon = std::numeric_limits<double>::epsilon();

    // Create a simple RNG for a uniform [-2, 10] distribution
    std::uniform_real_distribution dist {-2.0, 10.0 + epsilon};
    std::minstd_rand0 rng {};

    // An algorithm to generate numbers
    auto gen = [&]()
    {
        return dist(rng);
    };

    // An algorithm to generate a point in a function
    auto gen_point = [&](auto f)
    {
        auto x = gen();
        auto y = f(x);

        return Vec2 {x, y};
    };

    // An algorithm to generate a training point
    auto gen_train_point = [&]()
    {
        // As prescribed, the point satisfies y = x^2 + 10
        return gen_point([](auto x)
        {
            return x * x + 10;
        });
    };

    // An algorithm to generate multiple training points
    auto gen_multi_train_points = [&](int num)
    {
        std::vector<Vec2> points(num);

        for (int i = 0; i < num; ++i)
        {
            points[i] = gen_train_point();
        }

        return points;
    };

    // Generate twelve training points
    auto training = gen_multi_train_points(12);

    // Print out training set for reference
    std::cout << "TRAINING SET (n = " << training.size() << ")\n";
    std::cout << "-----------------------------------------------------------------\n";

    for (int i = 0; i < training.size(); ++i)
    {
        std::cout << (i + 1) << ". " << training[i] << "\n";
    }

    std::cout << "\nTASK 1 - LINEAR REGRESSION\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::system("pause");

    // Simply do linear regression on entire training set
    {
        auto [w, mse] = linreg(training.begin(), training.end());
        std::cout << "\nRegression line: y = " << w.y << " * x + " << w.x << "\n";
    }

    // Task 2
    std::cout << "\nTASK 2 - RIDGE REGRESSION WITH REGULARIZATION AND 3-FOLD CV\n";
    std::cout << "-----------------------------------------------------------------\n";
    std::system("pause");

    // Shuffle the training set
    std::shuffle(training.begin(), training.end(), std::minstd_rand0 {});

    for (auto lambda : std::array {0.1, 1.0, 10.0, 100.0})
    {
        std::cout << "\nLet lambda = " << lambda << "\n";

        // The average weight vector for this lambda
        Vec2 lambda_weight {};

        // The average MSE for this lambda
        double lambda_mse = 0;

        // The average validation error for this lambda
        double lambda_valid = 0;

        // Fold parameters
        // Only multiples of the training set size are supported right now
        const auto folds = 3;
        const auto per_fold = training.size() / folds;

        // Iterate fold-by-fold
        for (int fold = 0; fold < folds; ++fold)
        {
            // Inclusive-exclusive boundary indices
            auto idx_begin = fold * per_fold;
            auto idx_end = (fold + 1) * per_fold;

            // The sub-training and validation sets
            // These each have the same type as the original training set
            decltype(training) sub_train(per_fold * (folds - 1));
            decltype(training) sub_valid(per_fold);

            // Fill sub-training set
            std::copy(training.begin(), training.begin() + idx_begin, sub_train.begin());
            std::copy(training.begin() + idx_end, training.end(), sub_train.begin());

            // Fill sub-valid set
            std::copy(training.begin() + idx_begin, training.begin() + idx_end, sub_valid.begin());

            // Do the regression for this fold configuration
            auto [fold_weight, fold_mse] = linreg(sub_train.begin(), sub_train.end(), lambda);

            // Add weight and MSE to running totals
            // We'll divide these down later
            lambda_weight += fold_weight;
            lambda_mse += fold_mse;

            // Perform validation on the regression-producted weight vector
            lambda_valid += validate(sub_valid.begin(), sub_valid.end(), fold_weight);
        }

        // Divide the accumulators
        // They are now averages as advertised
        lambda_weight /= folds;
        lambda_mse /= folds;
        lambda_valid /= folds;

        std::cout << "\nFor lambda = " << lambda << "\n";
        std::cout << " -> Regression line: y = " << lambda_weight.y << " * x + " << lambda_weight.x << "\n";
        std::cout << " -> In-sample error: " << lambda_mse << "\n";
        std::cout << " -> Validation error: " << lambda_valid << "\n";
    }

    return 0;
}
