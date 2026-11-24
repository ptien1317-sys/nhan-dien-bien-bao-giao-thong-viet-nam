import time
from datetime import datetime

# Từ điển đầy đủ 73 lớp biển báo trong Dataset UIT (QCVN 41:2019/BGTVT)
QCVN_73_DICT = {
    "DP-135": ("Hết tất cả các lệnh cấm", "Biển hết cấm", "Thông báo: Hết đoạn đường áp dụng các lệnh cấm trước đó."),
    "P-102": ("Cấm đi ngược chiều", "Biển báo cấm", "CẢNH BÁO KHẨN: Đường cấm đi ngược chiều!"),
    "P-103A": ("Cấm xe ô tô", "Biển báo cấm", "Chú ý: Đường cấm tất cả các loại xe ô tô!"),
    "P-103B": ("Cấm ô tô rẽ phải", "Biển báo cấm", "Chú ý: Cấm xe ô tô rẽ phải tại giao lộ này!"),
    "P-103C": ("Cấm ô tô rẽ trái", "Biển báo cấm", "Chú ý: Cấm xe ô tô rẽ trái tại giao lộ này!"),
    "P-104": ("Cấm xe mô tô (xe máy)", "Biển báo cấm", "Chú ý: Đường cấm xe mô tô đi vào!"),
    "P105": ("Cấm xe ô tô và xe mô tô", "Biển báo cấm", "Cảnh báo: Đường cấm cả xe ô tô và xe máy!"),
    "P-106A": ("Cấm xe ô tô tải", "Biển báo cấm", "Chú ý: Đường cấm xe ô tô tải!"),
    "P-106B": ("Cấm ô tô tải trên khối lượng quy định", "Biển báo cấm", "Chú ý: Cấm xe tải vượt quá tải trọng cho phép!"),
    "P-107": ("Cấm ô tô khách và ô tô tải", "Biển báo cấm", "Chú ý: Cấm cả xe khách và xe tải!"),
    "P-107A": ("Cấm xe ô tô khách", "Biển báo cấm", "Chú ý: Đường cấm xe ô tô chở khách!"),
    "P-111": ("Cấm xe ba bánh / xe gắn máy", "Biển báo cấm", "Chú ý: Đường cấm xe ba bánh / xe gắn máy!"),
    "P-112": ("Cấm người đi bộ", "Biển báo cấm", "Thông báo: Khu vực cấm người đi bộ."),
    "P-115": ("Hạn chế tải trọng toàn bộ xe", "Biển báo cấm", "Cảnh báo: Giới hạn tổng trọng lượng xe qua cầu/đường!"),
    "P-117": ("Hạn chế chiều cao xe", "Biển báo cấm", "CẢNH BÁO: Giới hạn chiều cao tĩnh không phía trước!"),
    "P-123A": ("Cấm rẽ trái", "Biển báo cấm", "CẢNH BÁO: Cấm rẽ trái tại giao lộ phía trước!"),
    "P-123B": ("Cấm rẽ phải", "Biển báo cấm", "CẢNH BÁO: Cấm rẽ phải tại giao lộ phía trước!"),
    "P-124A": ("Cấm quay đầu xe", "Biển báo cấm", "Chú ý: Khu vực cấm quay đầu xe!"),
    "P-124B": ("Cấm ô tô quay đầu xe", "Biển báo cấm", "Chú ý: Cấm xe ô tô quay đầu tại đây!"),
    "P-124C": ("Cấm rẽ trái và cấm quay đầu xe", "Biển báo cấm", "CẢNH BÁO: Cấm cả rẽ trái và quay đầu xe!"),
    "P-125": ("Cấm vượt", "Biển báo cấm", "Cảnh báo: Đoạn đường cấm các xe vượt nhau!"),
    "P-127": ("Tốc độ tối đa cho phép", "Giới hạn tốc độ", "CẢNH BÁO TỐC ĐỘ: Chú ý giữ dưới tốc độ tối đa trên biển!"),
    "P-128": ("Cấm sử dụng còi", "Biển báo cấm", "Thông báo: Khu vực cấm bóp còi."),
    "P-130": ("Cấm dừng xe và đỗ xe", "Biển báo cấm", "Chú ý: Đoạn đường cấm cả dừng xe và đỗ xe!"),
    "P-131A": ("Cấm đỗ xe", "Biển báo cấm", "Chú ý: Đoạn đường cấm đỗ xe!"),
    "P-137": ("Cấm rẽ trái và rẽ phải", "Biển báo cấm", "CẢNH BÁO: Cấm rẽ trái và cấm rẽ phải!"),
    "P-245A": ("Đi chậm (SLOW)", "Biển cảnh báo", "Khuyến cáo: Giảm tốc độ, đi chậm quan sát!"),
    "R-301C": ("Các xe chỉ được rẽ trái", "Biển hiệu lệnh", "Hiệu lệnh bắt buộc: Các xe chỉ được rẽ trái!"),
    "R-301D": ("Các xe chỉ được rẽ phải", "Biển hiệu lệnh", "Hiệu lệnh bắt buộc: Các xe chỉ được rẽ phải!"),
    "R-301E": ("Các xe chỉ được rẽ trái (sau giao lộ)", "Biển hiệu lệnh", "Hiệu lệnh bắt buộc: Chỉ được rẽ trái!"),
    "R-302A": ("Hướng phải đi vòng chướng ngại vật sang phải", "Biển hiệu lệnh", "Hiệu lệnh: Đi vòng chướng ngại vật về phía bên phải!"),
    "R-302B": ("Hướng phải đi vòng chướng ngại vật sang trái", "Biển hiệu lệnh", "Hiệu lệnh: Đi vòng chướng ngại vật về phía bên trái!"),
    "R-303": ("Nơi giao nhau chạy theo vòng xuyến", "Biển hiệu lệnh", "Hiệu lệnh: Nhường đường và chạy theo vòng xuyến!"),
    "R-305": ("Đường dành cho người đi bộ", "Biển hiệu lệnh", "Thông báo: Phần đường dành riêng cho người đi bộ."),
    "R-401": ("Bắt đầu đường ưu tiên", "Biển chỉ dẫn", "Chỉ dẫn: Bắt đầu đoạn đường ưu tiên."),
    "R-402": ("Hết đoạn đường ưu tiên", "Biển chỉ dẫn", "Chú ý: Hết đoạn đường ưu tiên, chú ý nhường đường!"),
    "R-403": ("Đường dành cho xe ô tô", "Biển chỉ dẫn", "Chỉ dẫn: Tuyến đường dành riêng cho xe ô tô."),
    "R-407A": ("Đường một chiều", "Biển chỉ dẫn", "Chỉ dẫn: Tuyến đường lưu thông một chiều."),
    "R-409": ("Chỗ quay xe", "Biển chỉ dẫn", "Chỉ dẫn: Vị trí được phép quay đầu xe."),
    "R-411": ("Hướng đi trên mỗi làn đường phải theo", "Biển chỉ dẫn", "Chú ý: Đi đúng làn đường theo mũi tên chỉ dẫn!"),
    "R-412B": ("Làn đường dành cho xe ô tô con", "Biển chỉ dẫn", "Chỉ dẫn: Phân làn đường dành cho xe con."),
    "R-423": ("Vị trí người đi bộ sang ngang", "Biển chỉ dẫn", "Chú ý: Điểm sang đường cho người đi bộ phía trước."),
    "R-425": ("Bệnh viện / Cơ sở y tế", "Biển chỉ dẫn", "Thông báo: Gần khu vực bệnh viện, đi nhẹ hạn chế còi."),
    "R-434": ("Bến xe buýt", "Biển chỉ dẫn", "Thông báo: Điểm dừng đón trả khách xe buýt."),
    "S-509A": ("Biển phụ thuyết minh biển chính", "Biển phụ", "Thông tin bổ sung từ biển báo phụ."),
    "W-201A": ("Chỗ ngoặt nguy hiểm vòng bên trái", "Biển nguy hiểm", "Cảnh báo: Phía trước có khúc cua nguy hiểm sang trái!"),
    "W-201B": ("Chỗ ngoặt nguy hiểm vòng bên phải", "Biển nguy hiểm", "Cảnh báo: Phía trước có khúc cua nguy hiểm sang phải!"),
    "W-202A": ("Nhiều chỗ ngoặt nguy hiểm liên tiếp (trái)", "Biển nguy hiểm", "Giảm tốc độ: Đoạn đường quanh co liên tiếp!"),
    "W-202B": ("Nhiều chỗ ngoặt nguy hiểm liên tiếp (phải)", "Biển nguy hiểm", "Giảm tốc độ: Đoạn đường quanh co liên tiếp!"),
    "W-203B": ("Đường bị thu hẹp về phía trái", "Biển nguy hiểm", "Chú ý: Lòng đường bị thu hẹp bên trái!"),
    "W-203C": ("Đường bị thu hẹp về phía phải", "Biển nguy hiểm", "Chú ý: Lòng đường bị thu hẹp bên phải!"),
    "W-205A": ("Đường giao nhau (ngã tư)", "Biển nguy hiểm", "Giảm tốc độ: Sắp đến ngã tư đường giao nhau cùng cấp!"),
    "W-205B": ("Đường giao nhau (ngã ba bên phải)", "Biển nguy hiểm", "Chú ý: Có đường nhánh giao nhau từ bên phải!"),
    "W-205D": ("Đường giao nhau chữ T", "Biển nguy hiểm", "Chú ý: Sắp đến ngã ba giao nhau chữ T!"),
    "W-207": ("Giao nhau với đường không ưu tiên", "Biển nguy hiểm", "Chú ý: Có đường nhánh không ưu tiên cắt ngang!"),
    "W-207A": ("Giao nhau với đường không ưu tiên (hai bên)", "Biển nguy hiểm", "Chú ý: Giao lộ với đường nhánh không ưu tiên!"),
    "W-207B": ("Giao nhau với đường không ưu tiên (bên phải)", "Biển nguy hiểm", "Chú ý: Có ngõ/đường nhánh bên phải!"),
    "W-207C": ("Giao nhau với đường không ưu tiên (bên trái)", "Biển nguy hiểm", "Chú ý: Có ngõ/đường nhánh bên trái!"),
    "W-208": ("Giao nhau với đường ưu tiên (Nhường đường)", "Biển nguy hiểm", "CẢNH BÁO: Sắp giao cắt đường ưu tiên, chú ý nhường đường!"),
    "W208": ("Giao nhau với đường ưu tiên (Nhường đường)", "Biển nguy hiểm", "CẢNH BÁO: Sắp giao cắt đường ưu tiên, chú ý nhường đường!"),
    "W-209": ("Giao nhau có tín hiệu đèn", "Biển nguy hiểm", "Chú ý: Sắp đến giao lộ có đèn tín hiệu giao thông!"),
    "W-210": ("Giao nhau với đường sắt có rào chắn", "Biển nguy hiểm", "CẢNH BÁO: Phía trước giao cắt đường sắt có rào chắn!"),
    "W-219": ("Dốc xuống nguy hiểm", "Biển nguy hiểm", "CẢNH BÁO: Đoạn đường xuống dốc cao nguy hiểm, về số thấp!"),
    "W-221B": ("Đường có sóng mấp mô / Gồ giảm tốc", "Biển nguy hiểm", "Giảm tốc độ: Phía trước có gồ giảm tốc!"),
    "W-224": ("Đường người đi bộ cắt ngang", "Biển nguy hiểm", "Giảm tốc độ: Nhường đường cho người đi bộ sang ngang!"),
    "W-225": ("Cảnh báo Trẻ em / Khu vực trường học", "Biển nguy hiểm", "GIẢM TỐC ĐỘ: Khu vực trường học, chú ý trẻ em qua đường!"),
    "W-226": ("Đường có người đi xe đạp cắt ngang", "Biển nguy hiểm", "Chú ý: Đoạn đường có xe đạp cắt ngang!"),
    "W-227": ("Công trường đang thi công", "Biển nguy hiểm", "CẢNH BÁO: Phía trước có công trường đang thi công!"),
    "W-233": ("Nguy hiểm khác", "Biển nguy hiểm", "Cảnh báo: Chú ý quan sát chướng ngại vật phía trước!"),
    "W-235": ("Đường đôi", "Biển nguy hiểm", "Chỉ dẫn: Bắt đầu đoạn đường đôi có dải phân cách."),
    "W-245A": ("Đi chậm (SLOW)", "Biển nguy hiểm", "Khuyến cáo: Đi chậm quan sát an toàn!"),
    "CHI-DAN-KHAC": ("Biển chỉ dẫn giao thông", "Biển chỉ dẫn", "Thông báo: Biển chỉ dẫn thông tin lộ trình."),
    "CHO": ("Khu vực Chợ / Đông dân cư", "Biển chỉ dẫn", "Giảm tốc độ: Khu vực chợ đông người qua lại!"),
}

class ITSAlertManager:
    def __init__(self, cooldown_sec: float = 3.0):
        self.cooldown_sec = cooldown_sec
        self.last_alert_time = {}

    def process_detection(self, class_id: int, label_name: str, confidence: float):
        now = time.time()
        norm_code = str(label_name).strip().upper().replace("_", "-").replace(".", "-")

        if norm_code in QCVN_73_DICT:
            vn_name, group, alert_msg = QCVN_73_DICT[norm_code]
        else:
            vn_name = f"Biển báo {label_name}"
            group = "Biển báo Giao thông VN"
            alert_msg = f"Cảnh báo: Phát hiện biển báo {label_name}!"

        last_time = self.last_alert_time.get(norm_code, 0.0)
        is_new_alert = (now - last_time) >= self.cooldown_sec
        if is_new_alert:
            self.last_alert_time[norm_code] = now

        return {
            "timestamp": datetime.now().strftime("%H:%M:%S"),
            "code": norm_code,
            "name": vn_name,
            "group": group,
            "confidence": round(confidence * 100, 1),
            "alert_msg": alert_msg,
            "is_new_alert": is_new_alert
        }
