import os
from openai import OpenAI
import json
from tqdm import tqdm
import re


data = json.load(open("data/input/report-voice-20250101.json"))[:100]
# Initialize OpenAI client
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


def call_api_facebook(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are a helpful assistant that label text similarity, your",
            },
            {
                "role": "user",
                "content": f"""
                Hãy điền tiếp vào prompt sau:
                Text: Mẹ và 2 pé iu
                Output: None

                Text: LỬA THANH XUÂN Ngọc sáng ngời trong đêm cao vời vợi Lửa thanh xuân rực cháy một khoảng trời Tuổi anh hùng chẳng lý nào thảnh thơi Bởi trái tim đều hướng về Tổ Quốc Đêm đầu thu cái ngột ngạt của đô thị như dừng lại trước cánh cổng chính cơn gió heo may nhẹ
                Output: LỬA THANH XUÂN;Lửa thanh xuân;trái tim đều hướng về Tổ Quốc

                Text: 😅 1 ngày nắng cuối tuần 😓
                Output: None

                Text: KẾT QUẢ SIÊU TÍNH TOÁN CỦA THIÊN THẦN NHỎ 4 TUỔI CHỈ SAU 10 PHÚT CÔ THẢO GIẢNG BÀI TRÊN LỚP  Đến với Sgroup Hà Đông việc Thuộc bài tại lớp là kết quả tất yếu khi con thấy việc học là một TRẢI NGHIỆM KHÁM PHÁ chứ KHÔNG phải là phải HỌC   cô Thảo tự hào
                Output: KẾT QUẢ SIÊU TÍNH TOÁN CỦA THIÊN THẦN NHỎ 4 TUỔI

                Text: Năm ngoái rồi còn đưa lên Thương lắm Đà Nẵng ơi Cố lên nhé
                Output: Thương lắm Đà Nẵng ơi

                Text: {text}
                Output: """,
            },
        ],
    )

    return completion.choices[0].message.content


def call_api_news_(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Hãy điền tiếp vào prompt sau:

                Text: QK2 – Sáng 22/6, Đảng ủy Quân sự tỉnh Phú Thọ tổ chức Hội nghị sơ kết giữa nhiệm kỳ thực hiện nghị quyết Đại hội Đảng bộ Quân sự tỉnh Phú Thọ lần thứ XVIII; ra Nghị quyết lãnh đạo thực hiện nhiệm vụ quân sự, quốc phòng và công tác xây dựng Đảng bộ 6 tháng cuối năm và Bộ CHQS tỉnh Phú Thọ sơ kết công tác quốc phòng địa phương 6 tháng đầu năm 2023,triển khai nhiệm vụ 6 tháng cuối năm 2023. Đồng chí Bùi Văn Quang, Phó Bí thư Tỉnh ủy, Chủ tịch UBND tỉnh Phú Thọ chủ trì hội nghị. Tham dự có Đại tá Nguyễn Như Bách, Phó Chủ nhiệm Chính trị Quân khu 2. Đại tá Nguyễn Minh Long, Phó Bí thư Thường trực Đảng ủy Quân sự tỉnh Phú Thọ, Chính ủy Bộ CHQS tỉnh báo cáo kết quả thực hiện Nghị quyết Đại hội Đảng bộ Quân sự tỉnh. Từ đầu nhiệm kỳ đến nay, Đảng ủy Quân sự tỉnh Phú Thọ đã lãnh đạo hoàn thành 10/10 mục tiêu, chỉ tiêu theo kế hoạch nửa đầu nhiệm kỳ đề ra, nhiều nhiệm vụ hoàn thành xuất sắc. Đối với kết quả lãnh đạo thực hiện nhiệm vụ quân sự, quốc phòng và kết quả nhiệm vụ quốc phòng địa phương 6 tháng đầu năm, Đảng ủy, Bộ CHQS tỉnh đã lãnh đạo, chỉ đạo các cơ quan, đơn vị triển khai đồng bộ, toàn diện, hiệu quả trên các mặt công tác. Nổi bật: thực hiện hiệu quả các mặt công tác phối hợp nắm chắc tình hình địa bàn, sẵn sàng chiến đấu, tuyển quân, tổ chức Lễ ra quân huấn luyện, luyện tập chuyển trạng thái sẵn sàng chiến đấu, bảo vệ Lễ Giỗ Tổ Hùng Vương và tuần văn hóa du lịch Đất Tổ cùng nhiều nhiệm vụ quan trong khác. Tại hội nghị, các đại biểu đã thảo luận dân chủ, thắng thắn, nghiêm túc, chất lượng. Thông qua thảo luận, hội nghị đã nhất trí cao với dự thảo báo cáo kiểm điểm giữa nhiệm kỳ 2020 – 2025; dự thảo Nghị quyết lãnh đạo thực hiện nhiệm vụ 6 tháng cuối năm 2023; báo cáo sơ kết công tác quân sự, quốc phòng địa phương 6 tháng đầu năm 2023. Phát biểu kết luận hội nghị, đồng chí Bùi Văn Quang khẳng định, nhiệm vụ quân sự, quốc phòng thời gian còn lại của nhiệm kỳ hết sức nặng nề, tình hình có nhiều diễn biến phức tạp, đòi hỏi trong công tác lãnh đạo, chỉ đạo về công tác quân sự, quốc phòng phải hết sức quan tâm. Đồng chí đề nghị, các Đảng ủy, Bộ CHQS tỉnh và các cơ quan đơn vị tiếp tục quán triệt, triển khai đồng bộ, quyết liệt, hiệu quả Nghị quyết Đại hội Đảng bộ các cấp; các chủ trương, đường lối của Đảng, chính sách, pháp luật Nhà nước; các chiến lược về quân sự, quốc phòng, an ninh; lãnh đạo, chỉ đạo hoàn thành các mục tiêu, chỉ tiêu Nghị quyết Đại hội Đảng bộ Quân sự tỉnh lần thứ XVIII, nhiệm kỳ 2020 – 2025 đã xác định. Chú trọng chỉ đạo thực hiện những nhiệm vụ trọng tâm, 3 khâu đột… Đồng chí Chủ tịch UBND tỉnh cũng yêu cầu, ngay sau hội nghị, các cấp uỷ, tổ chức đảng, chỉ huy các cấp khẩn trương phổ biến, quán triệt, triển khai thực hiện tốt nhiệm vụ quân sự, quốc phòng địa phương 6 tháng cuối năm 2023; căn cứ vào chức năng, nhiệm vụ và tình hình thực tiễn địa phương, đơn vị mình xây dựng kế hoạch, triển khai thực hiện hiệu quả, sát thực tiễn. Tin, ảnh: TRỌNG LỘC
                Output: <kp>Đảng ủy Quân sự tỉnh Phú Thọ<sep>Hội nghị sơ kết giữa nhiệm kỳ<sep>nghị quyết Đại hội Đảng bộ Quân sự tỉnh Phú Thọ lần thứ XVII<sep>Đại tá Nguyễn Như Bách<sep>Đại tá Nguyễn Minh Long<sep>hoàn thành 10/10 mục tiêu<sep>Lễ Giỗ Tổ Hùng Vương<sep>tuần văn hóa du lịch Đất Tổ<sep>Đại hội Đảng bộ Quân sự tỉnh lần thứ XVIII</kp>

                Text: Ngày 29/6, lãnh đạo UBND phường Phước Hòa, TP Nha Trang, thông tin địa phương đã tiếp nhận phản ánh của một phụ huynh về việc con trai (học lớp 1) bị người dạy trẻ đánh.\nBước đầu, phường đã phối hợp công an địa phương kiểm tra cơ sở dạy học của bà H.P trên đường Chương Dương (phường Phước Hòa) và làm việc với người này sau khi nhận được phản ánh của phụ huynh. Theo đó, cơ sở dạy trẻ của bà H.P không được cấp thẩm quyền cấp phép.\nChị L.Q.P - mẹ cháu bé, cho biết do con trai học chậm nên chị đã lên mạng xã hội tìm lớp học dịp hè. Sau đó, chị cho con học lớp của bà H.P. Khi con về nhà, chị phát hiện con chảy máu ở miệng, mông thâm đen, đầu bị thương.\nChị gặng hỏi và bé cho biết đã bị bà H.P dùng gậy gỗ đánh do không thuộc bài. “Tôi đã trình báo công an, đồng thời giám định thương tật cho bé. Đến nay, bà H.P mới gọi điện xin lỗi và nói trả lại tiền 2 ngày bé chưa học”, chị Phương nói. Người mẹ này bày tỏ muốn cơ quan chức năng vào cuộc làm rõ vụ việc.\nLiên quan đến sự việc trên, đại diện Phòng GD-ĐT TP Nha Trang cho biết đang liên hệ chính quyền cấp phường để nắm tình hình và có hướng xử lý phù hợp.\n Xuân Ngọc  
                Output: <kp>con trai (học lớp 1) bị người dạy trẻ đánh<sep>công an địa phương kiểm tra cơ sở dạy học của bà H.P<sep>cơ sở dạy trẻ của bà H.P không được cấp thẩm quyền cấp phép</kp>

                Text: 2 chủ tài khoản facebook vi phạm gồm: bà T.T.L.C (50 tuổi, ngụ huyện Hóc Môn) và bà T.H.L (47 tuổi, ngụ quận 12). \nKhi bị mời làm việc, bà C. và bà L. đều nhận thức được việc đăng tải thông tin sai sự thật của bản thân, gây ảnh hưởng đến tình hình an ninh trật tự tại địa phương; đồng thời tự nguyện gỡ bỏ những bài viết vi phạm. \n \nCông an mời làm việc với bà T.H.L. Ảnh: CA \n \n \nVà làm việc với bà T.T.L.C. Ảnh: CA \nCơ quan chức năng xác định, hành vi của 2 cá nhân C. và L. đã vi phạm Điểm a và Điểm d, Khoản 1, Điều 101, Nghị định số 15/2020/NĐ-CP quy định xử phạt vi phạm hành chính trong lĩnh vực bưu chính viễn thông, tần số vô tuyến điện, công nghệ thông tin và giao dịch điện tử (sửa đổi, bổ sung tại Nghị định số 14/2022/NĐ-CP). \nCăn cứ mức độ vi phạm, Phòng An ninh mạng và phòng, chống tội phạm sử dụng công nghệ cao - Công an TP.HCM ra quyết định xử phạt hành chính 2 phụ nữ này mỗi người 7,5 triệu đồng. \nTrước đó, ngày 15/6, Công an TP.HCM cũng mời làm việc và ra quyết định xử phạt hành chính ông N.H.A.D (56 tuổi, ngụ tại xã Bình Hưng, huyện Bình Chánh) 7,5 triệu đồng vì có hành vi tương tự. \n \nCông an TP.HCM mời làm việc và xử phạt ông N.H.A.D. Ảnh: CA \nCông an TP.HCM đưa ra khuyến cáo, người dân theo dõi thông tin liên quan tình hình an ninh trật tự trên địa bàn TP.HCM và địa phương khác qua các kênh thông tin chính thống; tuyệt đối không đăng tải, chia sẻ các thông tin sai sự thật, chưa được kiểm chứng lên mạng xã hội gây hoang mang dư luận, ảnh hưởng tình hình an ninh, trật tự. \nĐàm Đệ \n
                Output: <kp>bà C. và bà L. đều nhận thức được việc đăng tải thông tin sai sự thật<sep>Phòng An ninh mạng và phòng, chống tội phạm sử dụng công nghệ cao<sep>Công an TP.HCM<sep>xử phạt hành chính 2 phụ nữ này mỗi người 7,5 triệu đồng<sep>tuyệt đối không đăng tải, chia sẻ các thông tin sai sự thật</kp>

                Text: \"Đoàn xe quân sự của quân đội Mỹ đã tiến từ Iraq vào Syria qua trạm kiểm soát Al-Walid. Một ngày trước đó có tới 40 phương tiện đã tiến tới các cơ sở quân sự của Mỹ ở tỉnh Hasakah của Syria, nơi do các chiến binh người Kurd kiểm soát\" - Anadolu viết. \n \nẢnh: AFP. \nTheo các nguồn tin ở Syria, đoàn xe của quân đội Mỹ bao gồm xe bọc thép, xe chở đạn dược và xe chở nhiên liệu. Đây là các thiết bị dành cho quân đội Mỹ ở các khu vực Rmelan, Аl-Shaddadi. \nMục đích chính \n \nQuân đội Mỹ kiểm soát trái phép các vùng lãnh thổ phía bắc và đông bắc Syria thuộc các tỉnh Deir ez-Zor, Al-Hasakah và Raqqa, nơi có các mỏ dầu khí lớn nhất Syria. \nChính quyền Damascus đã nhiều lần gọi sự hiện diện của quân đội Mỹ trên lãnh thổ Syria là chiếm đóng và cướp biển nhà nước với mục đích công khai ăn cắp dầu mỏ. \nThep Sputnik \n
                Output: <kp>Đoàn xe quân sự của quân đội Mỹ đã tiến từ Iraq vào Syria qua trạm kiểm soát Al-Walid<sep>cơ sở quân sự của Mỹ ở tỉnh Hasakah của Syria<sep>chiến binh người Kurd<sep>quân đội Mỹ ở các khu vực Rmelan, Аl-Shaddadi<sep>Quân đội Mỹ kiểm soát trái phép các vùng lãnh thổ phía bắc và đông bắc Syria<sep>Chính quyền Damascus<sep>mục đích công khai ăn cắp dầu mỏ</kp>

                Text: {text}
                Output: """,
            },
        ],
    )

    return completion.choices[0].message.content


def call_api_news(text):
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "user",
                "content": f"""
                Hãy bóc tách các keyphrase của bài viết sau; thường 1 keyphrase khoảng 4-9 từ là các cụm từ trong bài giúp tóm lược nội dung chính một cách ngắn gọn nhưng vẫn đầy đủ ý nghĩa, lưu ý keyphrase phải lấy từ text theo hướng extractive: không được thay đổi nội dung keyphrase so với trong text. Tôi cần bạn trả về các keyphrase được kẹp trong tag <kp> và các keyphrase được ngăn cách bởi token <sep>. VD: <kp>keyphrase1<sep>keyphrase2<sep>keyphrase3<sep>keyphrase4</kp>
                Đây là text: {text}
                """,
            },
        ],
    )

    return completion.choices[0].message.content


for row in tqdm(data[20_000:40_000]):
    text = row["text"][:1024]
    result = generate_fb(text=text, client=client, verbose=False)

    # out = copy.deepcopy(row)
    out = {}
    out["text"] = text
    out["label"] = extracted_label(result).split("<sep>")
    out["label"] = [x.strip() for x in out["label"]]
    out["label"] = [x for x in out["label"] if text.find(x) != -1]
    out["label"] = list(set(out["label"]))
    # out["label"] = [x for x in out["label"] if len(x) >= 5]

    json_str = json.dumps(out, ensure_ascii=False)

    with open("data/facebook/facebook_300k_2025_03/fb_50k_long_label.jsonl", "a") as f:
        f.write(json_str + "\n")
