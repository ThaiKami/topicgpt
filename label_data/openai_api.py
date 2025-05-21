from openai import OpenAI
import json
import os
import re

pattern = re.compile(r"<explaination>:\s*(.*?)\s*<answer>\s*([^\s]+)", re.DOTALL)
api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=api_key)
with open("label_data/pairwise_prompt.txt") as f:
    prompt_template = f.read()


def llm_pair_label(text1: str, text2: str) -> int:
    prompt = prompt_template.format(
        text1=text1,
        text2=text2,
    )
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
    )
    ans = resp.choices[0].message.content
    match = pattern.search(ans)
    if match:
        explanation = match.group(1).strip()
        answer = match.group(2).strip()
        print("Explanation:", explanation)
        print("Answer:", answer)
    else:
        print("No match found")
    return 1 if answer == "yes" else 0


# Example:
print(
    llm_pair_label(
        "Khách hàng đã gọi đến tổng đài VinaPhone để kiểm tra tình trạng gói cước internet 3G mà anh ta đã đăng ký trước đó. Anh ta muốn biết nếu anh ta đã hủy gói cước này chưa. Tổng đài viên xác nhận rằng gói cước đã được hủy.\nNội dung quan trọng:\n1. Khách hàng gọi đến tổng đài VinaPhone để kiểm tra tình trạng gói cước internet 3G mà anh ta đã đăng ký.\n2. Khách hàng muốn xác nhận nếu anh ta đã hủy gói cước này.\n3. Tổng đài viên xác nhận rằng gói cước đã được hủy.\n4. Khách hàng không còn sử dụng gói cước 3G nào trên số điện thoại mà anh ta đang gọi từ.\nCâu hỏi liên quan:\n1. Gói cước internet 3G mà khách hàng đã đăng ký trước đó có phải đã được hủy chưa?\n2. Khách hàng hiện tại có đang sử dụng gói cước 3G nào trên số điện thoại mà anh ta đang gọi từ không?",
        "Khách hàng muốn chuyển dịch vụ wifi từ địa chỉ cũ tại thôn 5, xã Tân Lĩnh, Yên Bái đến địa chỉ mới cũng tại thôn 5, cách địa chỉ cũ khoảng 2km. Tổng đài đã yêu cầu thông tin chi tiết và cam kết sẽ hỗ trợ việc chuyển dịch vụ, không tính phí nếu di dời trong phạm vi bình thường.\nNội dung quan trọng:\n1. Khách hàng muốn di dời dịch vụ wifi do nhà bị di dời do đất lở.\n2. Địa chỉ cũ: Thôn 5, xã Tân Lĩnh, Yên Bái.\n3. Địa chỉ mới: Cũng tại thôn 5, cách địa chỉ cũ khoảng 2km.\n4. Tổng đài yêu cầu khách hàng cung cấp thông tin chi tiết và cung cấp số liên hệ.\n5. Khách hàng mong muốn dịch vụ được chuyển nhanh nhất có thể, tốt nhất là vào ngày hôm sau.\n6. Tổng đài cam kết hỗ trợ việc di dời và không tính phí nếu di dời trong phạm vi bình thường.\n7. Khách hàng cần cung cấp căn cước công dân, giấy tờ địa chỉ mới và mang thiết bị đến để kỹ thuật hỗ trợ.\nCâu hỏi liên quan:\n1. Khách hàng có phải trả phí khi di dời dịch vụ wifi không?\n2. Thời gian khách hàng mong muốn dịch vụ được di dời là khi nào?\n3. Khách hàng cần cung cấp những thông tin gì để di dời dịch vụ?\n4. Tổng đài sẽ hỗ trợ khách hàng như thế nào trong việc di dời dịch vụ?",
    )
)  # → 1
