#include "adapt.hpp"

int main()
{
    ADAPT adapt;

    try
    {
        adapt.run();
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << std::endl;
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}