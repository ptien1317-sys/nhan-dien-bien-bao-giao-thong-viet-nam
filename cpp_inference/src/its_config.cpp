#include "its_config.hpp"

#include <fstream>
#include <map>
#include <stdexcept>

namespace its {
namespace {

std::string trim(const std::string& s) {
    const char* ws = " \t\r\n";
    const auto b = s.find_first_not_of(ws);
    if (b == std::string::npos) return "";
    return s.substr(b, s.find_last_not_of(ws) - b + 1);
}

// Bo chu thich '#' nam ngoai dau nhay
std::string stripComment(const std::string& line) {
    bool in_quote = false;
    char q = 0;
    for (size_t i = 0; i < line.size(); ++i) {
        const char c = line[i];
        if (in_quote) { if (c == q) in_quote = false; }
        else if (c == '"' || c == '\'') { in_quote = true; q = c; }
        else if (c == '#') return line.substr(0, i);
    }
    return line;
}

std::string unquote(std::string v) {
    v = trim(v);
    if (v.size() >= 2 && ((v.front() == '"' && v.back() == '"') || (v.front() == '\'' && v.back() == '\'')))
        return v.substr(1, v.size() - 2);
    return v;
}

// Bo doc YAML toi gian cho file cau hinh 2 cap:  "muc:" roi cac dong thut le "khoa: gia tri".
// Ket qua luu duoi dang map "muc.khoa" -> chuoi gia tri.
std::map<std::string, std::string> parseYaml(const std::string& path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error("Khong mo duoc file cau hinh: " + path);
    std::map<std::string, std::string> kv;
    std::string line, section;
    while (std::getline(f, line)) {
        line = stripComment(line);
        if (trim(line).empty()) continue;
        const bool indented = (line[0] == ' ' || line[0] == '\t');
        const auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        const std::string key = trim(line.substr(0, colon));
        const std::string val = trim(line.substr(colon + 1));
        if (!indented) {
            if (val.empty()) section = key;        // bat dau mot muc moi
            else kv[key] = unquote(val);           // khoa o cap goc
        } else {
            kv[section + "." + key] = unquote(val);
        }
    }
    return kv;
}

using Map = std::map<std::string, std::string>;
void getStr(const Map& m, const std::string& k, std::string& out) {
    auto it = m.find(k); if (it != m.end()) out = it->second;
}
void getInt(const Map& m, const std::string& k, int& out) {
    auto it = m.find(k);
    if (it == m.end() || it->second.empty()) return;
    try { out = std::stoi(it->second); }
    catch (...) { throw std::runtime_error("Gia tri khong phai so nguyen: " + k + " = " + it->second); }
}
void getFloat(const Map& m, const std::string& k, float& out) {
    auto it = m.find(k);
    if (it == m.end() || it->second.empty()) return;
    try { out = std::stof(it->second); }
    catch (...) { throw std::runtime_error("Gia tri khong phai so thuc: " + k + " = " + it->second); }
}
void getBool(const Map& m, const std::string& k, bool& out) {
    auto it = m.find(k);
    if (it == m.end()) return;
    const std::string& v = it->second;
    out = (v == "1" || v == "true" || v == "True");
}

}  // namespace

RuntimeConfig RuntimeConfig::load(const std::string& yaml_path) {
    const Map m = parseYaml(yaml_path);
    RuntimeConfig c;
    getStr(m, "model.path", c.model_path);
    getStr(m, "model.labels", c.labels_path);
    getInt(m, "model.input_width", c.input_width);
    getInt(m, "model.input_height", c.input_height);

    getFloat(m, "detection.conf_threshold", c.conf_threshold);
    getFloat(m, "detection.nms_threshold", c.nms_threshold);
    getInt(m, "detection.max_detections", c.max_detections);

    getStr(m, "backend.engine", c.engine);
    getStr(m, "backend.target", c.target);
    getInt(m, "backend.num_threads", c.num_threads);

    getFloat(m, "alert.conf_threshold", c.alert_conf_threshold);
    getInt(m, "alert.cooldown_ms", c.alert_cooldown_ms);
    getInt(m, "alert.banner_ms", c.alert_banner_ms);

    getStr(m, "source.input", c.source);
    getBool(m, "source.display", c.display);
    getStr(m, "source.output_video", c.output_video);

    getInt(m, "benchmark.warmup_frames", c.warmup_frames);
    getInt(m, "benchmark.max_frames", c.max_frames);
    getStr(m, "benchmark.result_json", c.result_json);
    getStr(m, "benchmark.tag", c.tag);
    return c;
}

}  // namespace its
