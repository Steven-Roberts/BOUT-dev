#include "gtest/gtest.h"
#include <bout/mesh.hxx>
#include "bout/field2d.hxx"
#include "bout/field3d.hxx"

class NVectorFixture : public ::testing::Test {
public:
  Mesh mesh;
  Field2D field1;

  NVectorFixture() : field1{0} {}
}

TEST_F(NVectorFixture, Constructor) {

}