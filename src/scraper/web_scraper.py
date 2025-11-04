"""
Web爬虫模块 - 使用Python进行网页数据采集
"""
import requests
from bs4 import BeautifulSoup
import time
import logging
from typing import List, Dict, Optional
from pathlib import Path
import json
from urllib.parse import urljoin, urlparse
import random

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    """Web爬虫类，提供网页数据采集功能"""
    
    def __init__(self, delay: float = 1.0, timeout: int = 10, max_retries: int = 3):
        """
        初始化爬虫
        
        Args:
            delay: 请求延迟（秒）
            timeout: 请求超时时间（秒）
            max_retries: 最大重试次数
        """
        self.delay = delay
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        })
        self.scraped_urls = set()
        
    def fetch_page(self, url: str) -> Optional[requests.Response]:
        """
        获取网页内容
        
        Args:
            url: 目标URL
            
        Returns:
            Response对象或None
        """
        for attempt in range(self.max_retries):
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                logger.info(f"成功获取页面: {url}")
                time.sleep(self.delay)
                return response
            except requests.exceptions.RequestException as e:
                logger.warning(f"请求失败 (尝试 {attempt + 1}/{self.max_retries}): {url} - {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.delay * (attempt + 1))
                else:
                    logger.error(f"最终失败: {url}")
                    return None
    
    def parse_html(self, html_content: str) -> BeautifulSoup:
        """
        解析HTML内容
        
        Args:
            html_content: HTML字符串
            
        Returns:
            BeautifulSoup对象
        """
        return BeautifulSoup(html_content, 'html.parser')
    
    def extract_text(self, soup: BeautifulSoup, 
                    selector: Optional[str] = None) -> str:
        """
        提取文本内容
        
        Args:
            soup: BeautifulSoup对象
            selector: CSS选择器（可选）
            
        Returns:
            提取的文本
        """
        if selector:
            elements = soup.select(selector)
            return ' '.join([elem.get_text(strip=True) for elem in elements])
        else:
            return soup.get_text(strip=True)
    
    def extract_links(self, soup: BeautifulSoup, 
                     base_url: str,
                     filter_domain: Optional[str] = None) -> List[str]:
        """
        提取链接
        
        Args:
            soup: BeautifulSoup对象
            base_url: 基础URL
            filter_domain: 过滤域名（可选）
            
        Returns:
            链接列表
        """
        links = []
        for link in soup.find_all('a', href=True):
            url = urljoin(base_url, link['href'])
            parsed = urlparse(url)
            
            if filter_domain:
                if parsed.netloc == filter_domain or parsed.netloc.endswith(f'.{filter_domain}'):
                    links.append(url)
            else:
                links.append(url)
        
        return list(set(links))
    
    def extract_images(self, soup: BeautifulSoup, 
                      base_url: str) -> List[Dict[str, str]]:
        """
        提取图片信息
        
        Args:
            soup: BeautifulSoup对象
            base_url: 基础URL
            
        Returns:
            图片信息列表
        """
        images = []
        for img in soup.find_all('img'):
            img_url = urljoin(base_url, img.get('src', ''))
            images.append({
                'url': img_url,
                'alt': img.get('alt', ''),
                'title': img.get('title', '')
            })
        return images
    
    def scrape(self, url: str, 
              extract_text: bool = True,
              extract_links: bool = False,
              extract_images: bool = False) -> Dict:
        """
        爬取网页数据
        
        Args:
            url: 目标URL
            extract_text: 是否提取文本
            extract_links: 是否提取链接
            extract_images: 是否提取图片
            
        Returns:
            提取的数据字典
        """
        if url in self.scraped_urls:
            logger.info(f"URL已爬取，跳过: {url}")
            return {}
        
        # 注意：self.fetch_page(url)方法仅通过requests库进行HTTP请求，无法获得经过JavaScript动态渲染的完整DOM树内容。
        # 如果目标网页内容是由JavaScript生成的（例如部分元素在XHR或AJAX后才出现在页面），则这里获取到的response.text通常只包含初始HTML，
        # 不包括JS运行后生成的内容。要抓取完整渲染后的页面，可以使用Selenium、Playwright等浏览器自动化工具。
        response = self.fetch_page(url)
        if not response:
            return {}
        
        soup = self.parse_html(response.text)
        data = {
            'url': url,
            'title': soup.title.string if soup.title else ''
        }
        
        if extract_text:
            data['text'] = self.extract_text(soup)
        
        if extract_links:
            data['links'] = self.extract_links(soup, url)
        
        if extract_images:
            data['images'] = self.extract_images(soup, url)
        
        self.scraped_urls.add(url)
        return data
    
    def scrape_multiple(self, urls: List[str], **kwargs) -> List[Dict]:
        """
        批量爬取多个URL
        
        Args:
            urls: URL列表
            **kwargs: 传递给scrape方法的参数
            
        Returns:
            数据列表
        """
        results = []
        for url in urls:
            data = self.scrape(url, **kwargs)
            if data:
                results.append(data)
        return results
    
    def save_data(self, data: List[Dict], file_path: str):
        """
        保存爬取的数据
        
        Args:
            data: 数据列表
            file_path: 保存路径
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"数据已保存: {file_path}, 共 {len(data)} 条记录")


class DataCollector:
    """数据采集器，结合爬虫和数据处理"""
    
    def __init__(self, scraper: WebScraper):
        self.scraper = scraper
    
    def collect_and_process(self, urls: List[str], 
                           output_path: str) -> List[Dict]:
        """
        采集并处理数据
        
        Args:
            urls: URL列表
            output_path: 输出路径
            
        Returns:
            处理后的数据列表
        """
        # 爬取数据
        raw_data = self.scraper.scrape_multiple(
            urls, 
            extract_text=True,
            extract_links=True,
            extract_images=True
        )
        
        # 数据处理
        processed_data = []
        for item in raw_data:
            processed_item = {
                'url': item['url'],
                'title': item['title'],
                'text_length': len(item.get('text', '')),
                'links_count': len(item.get('links', [])),
                'images_count': len(item.get('images', []))
            }
            processed_data.append(processed_item)
        
        # 保存数据
        self.scraper.save_data(processed_data, output_path)
        
        return processed_data


if __name__ == "__main__":
    # 示例用法
    scraper = WebScraper(delay=1.0)
    
    # 爬取单个页面
    data = scraper.scrape(
        'https://www.baidu.com',
        extract_text=True,
        extract_links=True
    )
    
    print("爬取结果:", json.dumps(data, ensure_ascii=False, indent=2))

