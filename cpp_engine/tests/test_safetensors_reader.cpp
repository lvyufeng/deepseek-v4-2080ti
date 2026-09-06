#include "safetensors_reader.hpp"

#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

void require(bool cond, const std::string& msg) {
    if (!cond) throw std::runtime_error(msg);
}

void require_tensor(const pocket::SafeTensorsShard& shard, const std::string& name, pocket::SafeDType dtype, std::initializer_list<uint64_t> shape) {
    const auto* info = shard.find_tensor(name);
    require(info != nullptr, "missing tensor: " + name);
    require(info->dtype == dtype, "bad dtype for " + name);
    require(info->shape == std::vector<uint64_t>(shape), "bad shape for " + name);
    require(shard.tensor_data(*info) != nullptr, "null tensor data: " + name);
}

void check_u8_payload_roundtrip() {
    const std::string path = "/tmp/qwen_u8_reader_roundtrip.safetensors";
    const std::string header =
        "{\"packed\":{\"dtype\":\"U8\",\"shape\":[2,3],"
        "\"data_offsets\":[0,6]}}";
    const uint64_t header_size = header.size();
    const uint8_t payload[6] = {0x10, 0x32, 0x54, 0x76, 0x98, 0xba};
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    require(static_cast<bool>(output), "cannot create U8 round-trip fixture");
    output.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    output.write(header.data(), static_cast<std::streamsize>(header.size()));
    output.write(reinterpret_cast<const char*>(payload), sizeof(payload));
    output.close();
    pocket::SafeTensorsShard shard(path);
    const auto* info = shard.find_tensor("packed");
    require(info != nullptr && info->dtype == pocket::SafeDType::U8 &&
                info->shape == std::vector<uint64_t>({2, 3}),
            "U8 round-trip metadata");
    require(std::memcmp(shard.tensor_data(*info), payload, sizeof(payload)) == 0,
            "U8 round-trip payload");
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: test_safetensors_reader <ckpt_dir>\n";
        return 2;
    }
    try {
        require(pocket::safe_dtype_from_string("U8") == pocket::SafeDType::U8,
                "U8 dtype parsing");
        require(pocket::safe_dtype_name(pocket::SafeDType::U8) == "U8",
                "U8 dtype naming");
        require(pocket::safe_dtype_size(pocket::SafeDType::U8) == 1,
                "U8 dtype size");
        check_u8_payload_roundtrip();
        std::string ckpt = argv[1];
        pocket::SafeTensorsIndex index(ckpt);
        require(index.tensor_count() == 69187, "unexpected tensor count");
        require(index.shard_count() == 46, "unexpected shard count");
        require(index.shard_for_tensor("embed.weight") != nullptr, "embed.weight missing from index");

        pocket::SafeTensorsShard shard1(index.shard_path("model-00001-of-00046.safetensors"));
        require_tensor(shard1, "embed.weight", pocket::SafeDType::BF16, {129280, 4096});

        pocket::SafeTensorsShard shard2(index.shard_path("model-00002-of-00046.safetensors"));
        require_tensor(shard2, "layers.0.ffn.gate.tid2eid", pocket::SafeDType::I64, {129280, 6});
        require_tensor(shard2, "layers.0.attn.wkv.weight", pocket::SafeDType::F8_E4M3, {512, 4096});
        require_tensor(shard2, "layers.0.attn.wkv.scale", pocket::SafeDType::F8_E8M0, {4, 32});

        std::cout << "[PASS] safetensors_reader tensors=" << index.tensor_count() << " shards=" << index.shard_count() << "\n";
        return 0;
    } catch (const std::exception& ex) {
        std::cerr << "[FAIL] " << ex.what() << "\n";
        return 1;
    }
}
