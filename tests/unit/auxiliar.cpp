#include "gtest/gtest.h"
#include "korali.hpp"
#include "auxiliar/jsonInterface.hpp"

namespace
{
 using namespace korali;

 TEST(Auxiliar, jsonInterface)
 {
  knlohmann::json js;

  ASSERT_NO_THROW(js["Key"] = "Value");
  ASSERT_NO_THROW(getValue(js, "Key"));
  ASSERT_NO_THROW(getPath("Key"));
  ASSERT_NO_THROW(getValue(js, "Unknown"));
 }

 TEST(Auxiliar, KoraliJson)
 {
  KoraliJson kjs;
  auto js = knlohmann::json();

  ASSERT_NO_THROW(kjs.setJson(js));
  ASSERT_NO_THROW(kjs["Key"] = "Value");
  ASSERT_NO_THROW(std::string v = kjs["Key"]);
 }

 TEST(Auxiliar, Math)
 {
  ASSERT_ANY_THROW(safeLogMinus(1.0, 2.0));
  ASSERT_NO_THROW(safeLogMinus(2.0, 1.0));
 }

} // namespace
