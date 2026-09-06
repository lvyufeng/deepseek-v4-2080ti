#pragma once

#include <cstdint>
#include <map>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace pocket {

struct JsonValue;
using JsonArray = std::vector<JsonValue>;
using JsonObject = std::map<std::string, JsonValue>;

struct JsonValue {
    using Value = std::variant<std::nullptr_t, bool, double, std::string, JsonArray, JsonObject>;
    Value value;

    bool is_null() const { return std::holds_alternative<std::nullptr_t>(value); }
    bool is_bool() const { return std::holds_alternative<bool>(value); }
    bool is_number() const { return std::holds_alternative<double>(value); }
    bool is_string() const { return std::holds_alternative<std::string>(value); }
    bool is_array() const { return std::holds_alternative<JsonArray>(value); }
    bool is_object() const { return std::holds_alternative<JsonObject>(value); }

    const JsonObject& object() const;
    const JsonArray& array() const;
    const std::string& string() const;
    double number() const;
    bool boolean() const;
};

JsonValue parse_json(const std::string& text);
const JsonValue* object_get(const JsonObject& obj, const std::string& key);
std::string json_required_string(const JsonObject& obj, const std::string& key);
uint64_t json_required_u64(const JsonObject& obj, const std::string& key);
std::vector<uint64_t> json_required_u64_array(const JsonObject& obj, const std::string& key);

}  // namespace pocket
