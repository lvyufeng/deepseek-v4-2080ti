#include "json_lite.hpp"

#include <cctype>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>

namespace pocket {

const JsonObject& JsonValue::object() const {
    if (!is_object()) throw std::runtime_error("JSON value is not an object");
    return std::get<JsonObject>(value);
}
const JsonArray& JsonValue::array() const {
    if (!is_array()) throw std::runtime_error("JSON value is not an array");
    return std::get<JsonArray>(value);
}
const std::string& JsonValue::string() const {
    if (!is_string()) throw std::runtime_error("JSON value is not a string");
    return std::get<std::string>(value);
}
double JsonValue::number() const {
    if (!is_number()) throw std::runtime_error("JSON value is not a number");
    return std::get<double>(value);
}
bool JsonValue::boolean() const {
    if (!is_bool()) throw std::runtime_error("JSON value is not a bool");
    return std::get<bool>(value);
}

namespace {

class Parser {
public:
    explicit Parser(const std::string& text) : text_(text) {}

    JsonValue parse() {
        skip_ws();
        JsonValue value = parse_value();
        skip_ws();
        if (pos_ != text_.size()) throw error("trailing characters");
        return value;
    }

private:
    JsonValue parse_value() {
        skip_ws();
        if (pos_ >= text_.size()) throw error("unexpected end");
        char c = text_[pos_];
        if (c == '{') return JsonValue{parse_object()};
        if (c == '[') return JsonValue{parse_array()};
        if (c == '"') return JsonValue{parse_string()};
        if (c == 't') return parse_literal("true", JsonValue{true});
        if (c == 'f') return parse_literal("false", JsonValue{false});
        if (c == 'n') return parse_literal("null", JsonValue{nullptr});
        if (c == '-' || std::isdigit(static_cast<unsigned char>(c))) return JsonValue{parse_number()};
        throw error("unexpected character");
    }

    JsonObject parse_object() {
        expect('{');
        JsonObject obj;
        skip_ws();
        if (consume('}')) return obj;
        while (true) {
            skip_ws();
            std::string key = parse_string();
            skip_ws();
            expect(':');
            obj.emplace(std::move(key), parse_value());
            skip_ws();
            if (consume('}')) break;
            expect(',');
        }
        return obj;
    }

    JsonArray parse_array() {
        expect('[');
        JsonArray arr;
        skip_ws();
        if (consume(']')) return arr;
        while (true) {
            arr.push_back(parse_value());
            skip_ws();
            if (consume(']')) break;
            expect(',');
        }
        return arr;
    }

    std::string parse_string() {
        expect('"');
        std::string out;
        while (pos_ < text_.size()) {
            char c = text_[pos_++];
            if (c == '"') return out;
            if (c == '\\') {
                if (pos_ >= text_.size()) throw error("bad escape");
                char e = text_[pos_++];
                switch (e) {
                    case '"': out.push_back('"'); break;
                    case '\\': out.push_back('\\'); break;
                    case '/': out.push_back('/'); break;
                    case 'b': out.push_back('\b'); break;
                    case 'f': out.push_back('\f'); break;
                    case 'n': out.push_back('\n'); break;
                    case 'r': out.push_back('\r'); break;
                    case 't': out.push_back('\t'); break;
                    case 'u': {
                        if (pos_ + 4 > text_.size()) throw error("bad unicode escape");
                        out.push_back('?');
                        pos_ += 4;
                        break;
                    }
                    default: throw error("unsupported escape");
                }
            } else {
                out.push_back(c);
            }
        }
        throw error("unterminated string");
    }

    double parse_number() {
        const char* begin = text_.c_str() + pos_;
        char* end = nullptr;
        double value = std::strtod(begin, &end);
        if (end == begin) throw error("bad number");
        pos_ += static_cast<size_t>(end - begin);
        return value;
    }

    JsonValue parse_literal(const char* literal, JsonValue value) {
        std::string s(literal);
        if (text_.compare(pos_, s.size(), s) != 0) throw error("bad literal");
        pos_ += s.size();
        return value;
    }

    void skip_ws() {
        while (pos_ < text_.size() && std::isspace(static_cast<unsigned char>(text_[pos_]))) ++pos_;
    }
    bool consume(char c) {
        if (pos_ < text_.size() && text_[pos_] == c) {
            ++pos_;
            return true;
        }
        return false;
    }
    void expect(char c) {
        if (!consume(c)) throw error(std::string("expected ") + c);
    }
    std::runtime_error error(const std::string& msg) const {
        std::ostringstream oss;
        oss << "JSON parse error at " << pos_ << ": " << msg;
        return std::runtime_error(oss.str());
    }

    const std::string& text_;
    size_t pos_ = 0;
};

}  // namespace

JsonValue parse_json(const std::string& text) {
    return Parser(text).parse();
}

const JsonValue* object_get(const JsonObject& obj, const std::string& key) {
    auto it = obj.find(key);
    return it == obj.end() ? nullptr : &it->second;
}

std::string json_required_string(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr) throw std::runtime_error("missing JSON string key: " + key);
    return value->string();
}

namespace {

uint64_t json_number_to_u64(double number, const std::string& key) {
    const double rounded = std::round(number);
    if (number < 0 || std::fabs(number - rounded) > 1e-9 || rounded > static_cast<double>(std::numeric_limits<uint64_t>::max())) {
        throw std::runtime_error("JSON number is not u64: " + key + " value=" + std::to_string(number));
    }
    return static_cast<uint64_t>(rounded);
}

}  // namespace

uint64_t json_required_u64(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr) throw std::runtime_error("missing JSON number key: " + key);
    return json_number_to_u64(value->number(), key);
}

std::vector<uint64_t> json_required_u64_array(const JsonObject& obj, const std::string& key) {
    const JsonValue* value = object_get(obj, key);
    if (value == nullptr) throw std::runtime_error("missing JSON array key: " + key);
    std::vector<uint64_t> out;
    for (const auto& item : value->array()) {
        out.push_back(json_number_to_u64(item.number(), key));
    }
    return out;
}

}  // namespace pocket
