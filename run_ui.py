#!/usr/bin/env python3
"""
多媒體圖像搜尋系統 - 啟動腳本
"""

import os
import sys

# 將當前目錄添加到路徑中
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# 導入UI模塊
from src.ui.app import ImageDatabaseUI

if __name__ == "__main__":
    # 處理命令行參數
    import argparse
    parser = argparse.ArgumentParser(description="多媒體圖像搜尋系統")
    parser.add_argument("--share", action="store_true", help="透過Gradio分享公開連結")
    parser.add_argument("--debug", action="store_true", help="啟用除錯模式")
    parser.add_argument("--port", type=int, default=None, help="指定運行的埠口 (預設自動尋找可用埠口)")
    args = parser.parse_args()
    
    print("="*50)
    print(" 多媒體圖像搜尋系統")
    print("="*50)
    print("初始化系統...")
    print(f"Python 路徑: {sys.path}")
    
    try:
        # 創建並啟動應用
        app = ImageDatabaseUI()
        port_display = args.port if args.port else "自動選擇"
        print(f"啟動網頁界面 (埠口: {port_display})")
        print(f"參數: share={args.share}, debug={args.debug}")
        app.launch(share=args.share, debug=args.debug, server_port=args.port)
    except KeyboardInterrupt:
        print("\n使用者已終止程式")
    except Exception as e:
        print(f"\n啟動時發生錯誤: {e}")
        import traceback
        traceback.print_exc()
    finally:
        print("\n感謝使用多媒體圖像搜尋系統!") 