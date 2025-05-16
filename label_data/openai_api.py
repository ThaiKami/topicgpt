import os
from openai import OpenAI


class deepseek:
    def __init__(self):
        # Load OpenAI API key and paths
        api_key = os.getenv("DEEPSEEK_API_KEY")
        self.name = "deepseek"
        # Initialize OpenAI client
        self.client = OpenAI(
            api_key=api_key,
            base_url="https://api.deepseek.com",
        )

    def get_completion(self, messages):
        completion = self.client.chat.completions.create(
            model="deepseek-chat",
            messages=messages,
            stream=False,
            top_p=0.0,
            temperature=0.0,
        )

        return completion.choices[0].message.content


class openai:
    def __init__(self):
        # Load OpenAI API key and paths
        api_key = os.getenv("OPENAI_API_KEY")
        self.name = "openai"
        # Initialize OpenAI client
        self.client = OpenAI(api_key=api_key)

    def get_completion(self, messages):
        completion = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            stream=False,
            top_p=0.0,
            temperature=0.0,
        )

        return completion.choices[0].message.content


# Initialize OpenAI client
client = OpenAI(api_key=api_key)
text = """Nữ chính khổ nhất phim Việt hiện tại: Nhập vai cực hay nhưng lại được chú ý vì nhan sắc Nữ diễn viên này có màn thể hiện đầy thuyết phục khi vào vai nhân vật bất hạnh. Hoa Hồng Cho Sớm Mai là bộ phim truyền hình hiện đang lên sóng trên kênh THVL. Phim đánh dấu sự tái xuất của nữ diễn viên nổi tiếng một thời - Tường Vi - với vai trò nữ chính bên cạnh dàn diễn viên đình đám khu vực phía Nam. Trong phim, Tường Vi vào vai Hằng, cô gái trải qua tuổi thơ bất hạnh, khốn khó. Cuộc sống "dễ thở" hơn khi bên cạnh Hằng có Thanh (Tim), thanh mai trúc mã gắn bó, chăm sóc cô từ ngày nhỏ. Hằng yêu Thanh bằng một tình yêu vô điều kiện, bất chấp thái độ ghẻ lạnh của bà Lê (Phương Dung), mẹ Thanh. Hạnh phúc mỉm cười khi Thanh quyết định cầu hôn Hằng. Thế nhưng biến cố xảy đến, Hằng vô tình gây tai nạn chết người, kết quả là phải ngồi tù 3 năm. Thanh giữ đúng lời hứa chờ Hằng ra tù nhưng giữa hai người dần có khoảng cách. Bà Lê thì liên tục ghẻ lạnh, hành hạ để Hằng tự biết đường tránh xa con mình. Bà thậm chí còn đẩy Hằng tới vũ trường, bắt cô làm tiếp rượu. Chịu đựng sự đày đoạ của mẹ chồng tương lai, Hằng vẫn nhẫn nhịn, cô thậm chí chấp nhận hạ thấp bản thân khi bàn tới tiếp sính lễ cưới hỏi để có thể ở bên Thanh. Trong khi đó Thanh lại thờ ơ, lạnh lùng với vợ sắp cưới. Đặc biệt, điều Hằng phải đối diện không chỉ là sự ghẻ lạnh, khinh miệt của người đời mà cô còn bị anh trai của nạn nhân 3 năm trước liên tục đeo bám để báo thù. Trung vốn có ý định đoạt mạng Hằng nhưng đã chuyển hướng, hành hạ để Hằng từ từ nếm trải đau khổ. Tất cả những điều này khiến cuộc sống của một cô gái lương thiện dần chìm trong bế tắc. Không ngoa khi nói Hằng chính là nữ chính khổ nhất phim Việt hiện tại. Theo dõi Hoa Hồng Cho Sớm Mai, khán giả dành rất nhiều lời khen cho màn thể hiện của Tường Vi. Vốn là một diễn viên có thực lực, vai Hằng không làm khó được cô. Những cảnh nặng cảm xúc, như phân đoạn Hằng nhận tin nạn nhân vụ tai nạn đã qua đời, được người xem đánh giá cao. Diễn hay là vậy nhưng khi Hoa Hồng Cho Sớm Mai lên sóng, thứ khán giả chú ý hơn cả lại là nhan sắc của Tường Vi. Để vào vai Hằng, cô phải xuất hiện trong tạo hình đơn giản, thậm chí là xuề xoà nhất có thể. Gương mặt gần như không trang điểm gì khiến các đường nét được phô ra một cách rõ nét nhất. Khán giả vui mừng khi gương mặt, nhất là cánh mũi từng bị biến dạng vì dao kéo của Tường Vi nay đã thanh thoát, tự nhiên hơn rất nhiều. Mặt mộc của Tường Vi được chú ý ở Hoa Hồng Cho Sớm Mai Gương mặt thanh thoát, tự nhiên hơn rất nhiều so với trước đây Nguồn ảnh: THVL
"""
completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "user",
            "content": f"""
                Hãy điền tiếp vào prompt sau:

            Text: ▁Chúc▁mừng▁Chị▁Đẹp▁Nguyen▁Ha▁Nguyên▁Hà▁với▁MV▁mới▁ra▁mắt▁“Như▁Lần▁Đầu”.▁Giọng▁hát▁trong▁trẻo▁khiến▁trái▁tim▁thổn▁thức,▁nổn▁nức,▁nhổn▁nhức.▁Em▁không▁hề▁seeding▁mà▁em▁nói▁thật▁từ▁tâm▁nha▁cả▁nhà.▁Cả▁nhà▁vào▁nghe▁mà▁xem▁em▁nói▁có▁điêu▁hay▁là
            Output: ▁“Như▁Lần▁Đầu”

            Text: Cục cưng diễn văn nghệ 20\/11\nNhư này là giỏi hơn Mẹ à hahaa\n\n
            Output: ▁văn▁nghệ▁20\/11

            Text: 😅 1 ngày nắng cuối tuần 😓\n\n
            Output: ▁1▁ngày▁nắng

            Text: ▁Đứa▁bạn▁vừa▁chạy▁xong▁suất▁kết▁hôn▁giả▁để▁đi▁Mỹ▁với▁giá▁50.000▁đô,▁hoàn▁thành▁giấc▁mơ▁trở▁thành▁công▁dân▁của▁đất▁nước▁cờ▁hoa▁mà▁nó▁ấp▁ủ▁bấy▁lâu▁nay.▁Nó▁phân▁bua▁với▁tôi,▁với▁ngần▁ấy▁tiền,▁ở▁Việt▁Nam▁tao▁được▁gọi▁là▁tỷ▁phú,▁có▁thể▁sống▁ph
            Output: ▁kết▁hôn▁giả▁để▁đi▁Mỹ▁với▁giá▁50.000▁đô

            Text: ▁😅😅▁Đã▁tìm▁ra▁bai▁hat▁hay▁nhất▁cho▁chi▁em▁Hết▁cưu▁rồi▁mấy▁anh▁chồng▁ơi
            Output: ▁bai▁hat▁hay▁nhất▁cho▁chi▁em

                Text: {text}
                Output: """,
        },
    ],
)

if __name__ == "__main__":
    output = generate_news(text=text, client=openai_client, verbose=True)
    print(output)
