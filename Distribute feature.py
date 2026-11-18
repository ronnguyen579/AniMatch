import pandas as pd
import requests
import time
import json

# ==========================================
# 1. HÀM GỌI API LẤY ĐẦY ĐỦ TÊN VÀ SẮP XẾP THEO NĂM
# ==========================================
def fetch_sorted_franchise(anime_title):
    # GraphQL Query: Thêm english và native vào object title
    query = '''
    query ($search: String) {
      Media (search: $search, type: ANIME) {
        id
        title { romaji english native }
        startDate { year }
        relations {
          edges {
            relationType
            node {
              id
              type
              title { romaji english native }
              startDate { year }
            }
          }
        }
      }
    }
    '''
    variables = {'search': anime_title}
    url = 'https://graphql.anilist.co'
    
    try:
        response = requests.post(url, json={'query': query, 'variables': variables})
        if response.status_code == 200:
            data = response.json().get('data', {}).get('Media')
            if not data:
                return None, [], []
            
            franchise_list = []
            
            # Hàm xử lý tên để tránh lỗi khi một số bộ không có tên Tiếng Anh hoặc Native
            def extract_titles(title_obj):
                romaji = title_obj.get('romaji') or "Unknown"
                return {
                    'romaji': romaji,
                    'english': title_obj.get('english') or romaji, # Fallback về romaji nếu không có tiếng Anh
                    'native': title_obj.get('native') or "Unknown"
                }

            # 1. Thêm bộ gốc vào danh sách
            root_year = data.get('startDate', {}).get('year')
            franchise_list.append({
                'id': data['id'],
                'titles': extract_titles(data['title']),
                'year': root_year if root_year else 9999
            })
            
            # 2. Quét qua các phim liên quan (Sequel, Prequel...)
            if data.get('relations') and data['relations'].get('edges'):
                for edge in data['relations']['edges']:
                    node = edge['node']
                    if node['type'] == 'ANIME':
                        node_year = node.get('startDate', {}).get('year')
                        franchise_list.append({
                            'id': node['id'],
                            'titles': extract_titles(node['title']),
                            'year': node_year if node_year else 9999
                        })
            
            # 3. Khử trùng lặp theo ID
            unique_franchise = {anime['id']: anime for anime in franchise_list}.values()
            
            # 4. Sắp xếp theo năm phát hành
            sorted_franchise = sorted(unique_franchise, key=lambda x: x['year'])
            
            # Trích xuất mảng ID và mảng thông tin chi tiết
            sorted_ids = [anime['id'] for anime in sorted_franchise]
            
            # Tạo chuỗi hiển thị chứa đủ 3 loại tên để kiểm tra dễ dàng hơn
            sorted_titles_info = []
            for anime in sorted_franchise:
                t = anime['titles']
                info = f"[{anime['year']}] {t['english']} | {t['native']} (Romaji: {t['romaji']})"
                sorted_titles_info.append(info)
            
            return data['id'], sorted_ids, sorted_titles_info
            
        elif response.status_code == 429:
            print("\n[!] Bị giới hạn rate limit. Đang đợi 5 giây...")
            time.sleep(5)
            return fetch_sorted_franchise(anime_title)
        else:
            print(f"Lỗi API {response.status_code}: {anime_title}")
            return None, [], []
            
    except Exception as e:
        print(f"Lỗi kết nối: {e}")
        return None, [], []


# ==========================================
# 2. ĐỌC FILE CSV VÀ ĐỒNG BỘ DỮ LIỆU
# ==========================================
csv_file_path = "anime_data.csv"
output_file = "anime_data_franchise_full_titles.csv"

try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Không tìm thấy {csv_file_path}. Tạo dữ liệu mẫu...")
    df = pd.DataFrame({
        "anime_id": [1, 2, 3],
        "title": ["Fate/stay night", "Kimetsu no Yaiba", "Shingeki no Kyojin"],
        "genres": ["Action Fantasy", "Action Historical", "Action Drama"]
    })

df['anilist_id'] = None
df['franchise_sorted_ids'] = None
df['franchise_chronological_info'] = None

print("\nBẮT ĐẦU ĐỒNG BỘ FRANCHISE VỚI ĐẦY ĐỦ TÊN...")

for index, row in df.iterrows():
    search_title = row['title']
    print(f"Đang xử lý: {search_title:<25}", end=" ")
    
    main_id, sorted_ids, sorted_titles_info = fetch_sorted_franchise(search_title)
    
    if main_id:
        df.at[index, 'anilist_id'] = main_id
        df.at[index, 'franchise_sorted_ids'] = json.dumps(sorted_ids)
        # Lưu mảng chứa chuỗi thông tin năm, tiếng Anh, tiếng Nhật, tiếng Romaji
        df.at[index, 'franchise_chronological_info'] = json.dumps(sorted_titles_info, ensure_ascii=False)
        print(f"-> Hoàn tất! ({len(sorted_ids)} phần)")
    else:
        print("-> Không tìm thấy trên AniList.")
        
    time.sleep(0.7)

df.to_csv(output_file, index=False)
print(f"\n[Thành công] Đã lưu dữ liệu cập nhật vào: {output_file}")


# ==========================================
# 3. KIỂM TRA DỮ LIỆU ĐẦU RA CỦA ANIME ĐẦU TIÊN
# ==========================================
print("\n--- MÔ PHỎNG DỮ LIỆU FRANCHISE HOÀN CHỈNH ---")
sample_anime = df.iloc[0]

if pd.notna(sample_anime['franchise_sorted_ids']):
    sorted_titles = json.loads(sample_anime['franchise_chronological_info'])
    
    print(f"Dòng thời gian của: {sample_anime['title']}")
    for idx, info in enumerate(sorted_titles):
        print(f"  {idx + 1}. {info}")