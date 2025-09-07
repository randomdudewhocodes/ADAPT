cmake_minimum_required(VERSION 3.10)
project(ADAPT)

set(CMAKE_CXX_STANDARD 17)

# --- Vulkan ---
find_package(Vulkan REQUIRED)
find_program(GLSLC_EXECUTABLE glslc REQUIRED)

# --- GLFW ---
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL OFF CACHE BOOL "" FORCE)
add_subdirectory(${CMAKE_SOURCE_DIR}/ext/glfw ${CMAKE_BINARY_DIR}/ext/glfw)

# --- Shaders ---
function(add_shaders TARGET_NAME)
    set(SHADER_SOURCE_FILES ${ARGN})
    list(LENGTH SHADER_SOURCE_FILES FILE_COUNT)

    if(FILE_COUNT EQUAL 0)
        file(GLOB SHADER_SOURCE_FILES
            "${CMAKE_SOURCE_DIR}/src/shaders/*.comp"
            "${CMAKE_SOURCE_DIR}/src/shaders/*.vert"
            "${CMAKE_SOURCE_DIR}/src/shaders/*.frag"
            "${CMAKE_SOURCE_DIR}/src/shaders/*.geom"
            "${CMAKE_SOURCE_DIR}/src/shaders/*.tesc"
            "${CMAKE_SOURCE_DIR}/src/shaders/*.tese"
            "${CMAKE_SOURCE_DIR}/src/shaders/*.glsl"
        )
    endif()

    if(NOT SHADER_SOURCE_FILES)
        message(FATAL_ERROR "No shader source files found")
    endif()

    set(SHADER_COMMANDS)
    set(SHADER_PRODUCTS)
    set(SHADER_OUTPUT_DIR "${CMAKE_SOURCE_DIR}/src/compiled shaders")

    foreach(SHADER_SOURCE IN LISTS SHADER_SOURCE_FILES)
        get_filename_component(SHADER_SOURCE_ABS "${SHADER_SOURCE}" ABSOLUTE)
        file(RELATIVE_PATH SHADER_SOURCE_REL "${CMAKE_SOURCE_DIR}" "${SHADER_SOURCE_ABS}")

        get_filename_component(SHADER_BASENAME "${SHADER_SOURCE_ABS}" NAME_WE)
        set(SHADER_OUTPUT_FILE "${SHADER_OUTPUT_DIR}/${SHADER_BASENAME}.spv")

        list(APPEND SHADER_COMMANDS
            COMMAND ${CMAKE_COMMAND} -E echo "Compiling: ${SHADER_SOURCE_REL}"
            COMMAND ${GLSLC_EXECUTABLE} "${SHADER_SOURCE_ABS}" -o "${SHADER_OUTPUT_FILE}"
        )

        list(APPEND SHADER_PRODUCTS "${SHADER_OUTPUT_FILE}")
    endforeach()

    add_custom_target(${TARGET_NAME} ALL
        COMMAND ${CMAKE_COMMAND} -E make_directory "${SHADER_OUTPUT_DIR}"
        ${SHADER_COMMANDS}
        COMMENT "Compiling Shaders [${TARGET_NAME}]"
        SOURCES ${SHADER_SOURCE_FILES}
        BYPRODUCTS ${SHADER_PRODUCTS}
    )
endfunction()

# --- Compile shaders ---
add_shaders(shaders)

# --- Executable ---
add_executable(${PROJECT_NAME} src/main.cpp src/adapt.cpp)

# --- Include paths ---
target_include_directories(
    ${PROJECT_NAME} PRIVATE ${Vulkan_INCLUDE_DIRS}
    ${CMAKE_SOURCE_DIR}/ext/glm
    ${CMAKE_SOURCE_DIR}/ext/stb
)

# --- Link libraries ---
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan glfw)
