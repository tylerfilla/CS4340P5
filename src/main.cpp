/*
 * Tyler Filla
 * CS 4340 - Project 5
 * December 13, 2018
 */

#include <array>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <iterator>
#include <limits>
#include <random>

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
 */
template<class InputIt>
static Vec2 linreg(InputIt begin, InputIt end, double lambda = 0)
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
        if (grad.dot(grad) <= th * th)
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

    return weight;
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
    std::cout << "--------------------------------------------------\n";
    for (int i = 0; i < training.size(); ++i)
    {
        std::cout << (i + 1) << ". " << training[i] << "\n";
    }

    std::system("pause");

    std::cout << "\nTASK 1 - LINEAR REGRESSION\n";
    std::cout << "--------------------------------------------------\n";

    // Simply do linear regression on entire training set
    {
        auto w = linreg(training.begin(), training.end());
        std::cout << "\nRegression line: y = " << w.y << " * x + " << w.x << "\n";
    }

    std::system("pause");

    // Task 2
    std::cout << "\nTASK 2 - RIDGE REGRESSION WITH REGULARIZATION\n";
    std::cout << "--------------------------------------------------\n";

    for (auto lambda : std::array {0.1, 1.0, 10.0, 100.0})
    {
        std::cout << "Let lambda = " << lambda << "\n";
        linreg(training.begin(), training.end(), lambda);
        std::cout << "\n";
    }

    return 0;
}
