/*
 * Tyler Filla
 * CS 4340 - Project 5
 * December 13, 2018
 */

#include <iostream>
#include <limits>
#include <random>

/**
 * A single 2D data point.
 */
struct Point
{
    double x;
    double y;
};

/**
 * Print a point to an output stream.
 *
 * @param out The output stream
 * @param pt The point
 * @return The output stream
 */
static std::ostream& operator<<(std::ostream& out, const Point& pt)
{
    return out << "(" << pt.x << ", " << pt.y << ")";
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

        return Point {x, y};
    };

    // An algorithm to generate a test point
    auto gen_test_point = [&]()
    {
        // As prescribed, the point satisfies y = x^2 + 10
        return gen_point([](auto x)
        {
            return x * x + 10;
        });
    };

    // An algorithm to generate multiple test points
    auto gen_multi_test_points = [&](int num)
    {
        std::vector<Point> points(num);
        for (int i = 0; i < num; ++i)
        {
            points[i] = gen_test_point();
        }
        return points;
    };

    // Generate twelve training points
    auto training = gen_multi_test_points(12);

    // Print out training set for reference
    std::cout << "TRAINING SET (n = " << training.size() << ")\n";
    std::cout << "------------------------------\n";
    for (int i = 0; i < training.size(); ++i)
    {
        std::cout << (i + 1) << ". " << training[i] << "\n";
    }

    // Task 1
    std::cout << "\nTASK 1 - LINEAR REGRESSION\n";
    // TODO

    // Task 2
    std::cout << "\nTASK 2 - RIDGE REGRESSION WITH GIVEN LAMBDAS\n";
    // TODO

    // Task 3
    std::cout << "\nTASK 3 - RIDGE REGRESSION WITH 3-FOLD CV\n";
    // TODO

    return 0;
}
