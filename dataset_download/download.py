import argparse, json, os, re, requests
from zipfile import ZipFile

ua = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/90.0.4430.85 Safari/537.36"

headers = {'User-agent': ua}
cookies = json.load(open('cookies.json', 'r', encoding='utf-8'))

def download_file(subject, part, download_dir, keep_zip):
    filename = f'DB6_s{subject}_{part}.zip'
    url = f'http://ninapro.hevs.ch/system/files/DB6_Preproc/{filename}'
    download_path = os.path.join(download_dir, filename)
    
    os.makedirs(download_dir, exist_ok=True)
    
    # https://stackoverflow.com/a/1094933
    def sizeof_fmt(num, suffix='B'):
        for unit in ['','Ki','Mi','Gi','Ti','Pi','Ei','Zi']:
            if abs(num) < 1024.0:
                return "%3.1f%s%s" % (num, unit, suffix)
            num /= 1024.0
        return "%.1f%s%s" % (num, 'Yi', suffix)
    
    if not os.path.isfile(download_path):
        with requests.get(url, headers=headers, cookies=cookies, stream=True) as r:
            r.raise_for_status()
            
            total_filesize_GiB = int(r.headers['Content-Length']) / (2 ** 30)
            print(f'File size: {total_filesize_GiB:.1f} GiB', flush=True)
            
            with open(download_path + '.part', 'wb') as f:
                tot_bytes_downloaded = 0
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
                    tot_bytes_downloaded += len(chunk)
                    print("Downloaded " + sizeof_fmt(tot_bytes_downloaded) + " " * 20 +"\r", end="", flush=True)
        os.rename(download_path + '.part', download_path)
        print("Downloaded in", download_path, flush=True)
    
    with ZipFile(download_path, 'r') as zipFile:
        sessions = [info for info in zipFile.infolist() if os.path.splitext(info.filename)[1] == '.mat']
        for session in sessions:
            session.filename = os.path.basename(session.filename)
            print("Extracting", session.filename)
            zipFile.extract(session, path=download_dir)
            
    if keep_zip == 'no':
        os.remove(download_path)
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subject', type=str, required=True, choices=list(map(str, range(1, 11))))
    parser.add_argument('--part', type=str, required=True, choices=('a', 'b'))
    parser.add_argument('--download-dir', type=str, default='.')
    parser.add_argument('--keep-zip', type=str, default='yes', choices=('yes', 'no'))
    download_file(**vars(parser.parse_args()))