#!/usr/bin/env python3
"""
多媒體圖像搜尋系統啟動器
啟動Gradio界面用於上傳、搜尋和查詢圖片
"""

import os
import sys

# 添加src目錄到路徑
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# 使用絕對路徑導入
from src.ui.app import ImageDatabaseUI

if __name__ == "__main__":
    # 處理命令行參數
    import argparse
    parser = argparse.ArgumentParser(description="多媒體圖像搜尋系統")
    parser.add_argument("--share", action="store_true", help="透過Gradio分享公開連結")
    parser.add_argument("--debug", action="store_true", help="啟用除錯模式")
    args = parser.parse_args()
    
    print("初始化多媒體圖像搜尋系統...")
    app = ImageDatabaseUI()
    
    print(f"啟動網頁界面 (share={args.share}, debug={args.debug})...")
    app.launch(share=args.share, debug=args.debug) 