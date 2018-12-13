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

    // FIXME
    for (auto&& pt : gen_multi_test_points(5))
    {
        std::cout << pt << "\n";
    }

    return 0;
}
