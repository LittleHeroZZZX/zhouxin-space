(() => {
    let activeElement;
    let elements;
    let tocLinks;
    let isScrolling = false;

    const main = parseInt(getComputedStyle(document.body).getPropertyValue('--article-width'), 10);
    const toc = parseInt(getComputedStyle(document.body).getPropertyValue('--toc-width'), 10);
    const gap = parseInt(getComputedStyle(document.body).getPropertyValue('--gap'), 10);

    document.addEventListener('DOMContentLoaded', init);

    function init() {
        checkTocPosition();
        setupElements();
        setupTopLink();
        setupTocLinks();

        window.addEventListener('resize', checkTocPosition);
        window.addEventListener('scroll', onScroll, { passive: true });
    }

    function setupElements() {
        elements = [...document.querySelectorAll('h1[id],h2[id],h3[id],h4[id],h5[id],h6[id]')];
        if (elements.length > 0) {
            activeElement = elements[0];
            const id = encodeURIComponent(activeElement.id).toLowerCase();
            document.querySelector(`.inner ul li a[href="#${id}"]`)?.classList.add('active');
        }
    }
    function scrollTocToTop() {
        const tocInner = document.querySelector('.toc .inner');
        if (tocInner) {
            tocInner.scrollTo({ top: 0, behavior: 'smooth' });
        }
    }

    function setupTopLink() {
        const topLink = document.getElementById('top-link');
        topLink?.addEventListener('click', (event) => {
            event.preventDefault();
            window.scrollTo({ top: 0, behavior: 'smooth' });
            scrollTocToTop();
            updateActiveElement(); // 更新活动元素
        });
    }

    function setupTocLinks() {
        tocLinks = [...document.querySelectorAll('.toc .inner ul li a')];
        tocLinks.forEach(link => {
            link.addEventListener('click', handleTocLinkClick);
        });
    }

    function handleTocLinkClick(event) {
        event.preventDefault();
        const href = event.currentTarget.getAttribute('href');
        const targetId = decodeURIComponent(href.substring(1));
        const targetElement = document.getElementById(targetId);

        if (targetElement) {
            isScrolling = true;
            window.removeEventListener('scroll', onScroll);

            smoothScrollToElement(targetElement).then(() => {
                isScrolling = false;
                window.addEventListener('scroll', onScroll, { passive: true });
            });
        }
    }

    async function smoothScrollToElement(element) {
        const start = window.pageYOffset;
        const end = getOffsetTop(element);
        const duration = 1000;
        const startTime = performance.now();

        while (performance.now() - startTime < duration) {
            const elapsed = performance.now() - startTime;
            const progress = Math.min(elapsed / duration, 1);
            const easeInOutCubic = progress < 0.5 
                ? 4 * progress ** 3 
                : 1 - Math.pow(-2 * progress + 2, 3) / 2;

            window.scrollTo(0, start + (end - start) * easeInOutCubic);
            await new Promise(requestAnimationFrame);

            updateActiveElement();
        }

        window.scrollTo(0, end);
    }

    function onScroll() {
        if (isScrolling) return;
        updateActiveElement();
    }

    function updateActiveElement() {
        if (!elements || elements.length === 0) return;
    
        const scrollPosition = window.pageYOffset || document.documentElement.scrollTop;
        const windowHeight = window.innerHeight;
        const documentHeight = Math.max(
            document.body.scrollHeight, document.documentElement.scrollHeight,
            document.body.offsetHeight, document.documentElement.offsetHeight,
            document.body.clientHeight, document.documentElement.clientHeight
        );
    
        // 检查是否在页面顶部
        if (scrollPosition === 0) {
            if (activeElement !== elements[0]) {
                activeElement = elements[0];
                updateTocLinks();
            }
            return;
        }
    
        // 检查是否滚动到了页面底部
        const isAtBottom = scrollPosition + windowHeight >= documentHeight - 50; // 50px 的缓冲区
    
        let newActiveElement;
    
        if (isAtBottom) {
            // 如果在底部，选择最后一个元素
            newActiveElement = elements[elements.length - 1];
        } else {
            // 否则，查找视窗中的第一个元素
            newActiveElement = elements.find((element) => {
                const elementTop = getOffsetTop(element);
                return (elementTop - scrollPosition) > 0 && 
                        (elementTop - scrollPosition) < windowHeight / 2;
            });
        }
    
        // 如果没有找到新的活动元素，保持当前活动元素不变
        if (!newActiveElement) {
            return;
        }
    
        if (newActiveElement !== activeElement) {
            activeElement = newActiveElement;
            updateTocLinks();
        }
    }
    

    function updateTocLinks() {
        tocLinks.forEach(link => {
            const id = encodeURIComponent(activeElement.id).toLowerCase();
            if (link.getAttribute('href') === `#${id}`) {
                link.classList.add('active');
            } else {
                link.classList.remove('active');
            }
        });
    }

    function checkTocPosition() {
        const width = document.body.scrollWidth;
        const tocContainer = document.getElementById("toc-container");

        if (width - main - (toc * 2) - (gap * 4) > 0) {
            tocContainer?.classList.add("wide");
        } else {
            tocContainer?.classList.remove("wide");
        }
    }

    function getOffsetTop(element) {
        if (!element.getClientRects().length) return 0;
        const rect = element.getBoundingClientRect();
        const win = element.ownerDocument.defaultView;
        return rect.top + win.pageYOffset;   
    }
})();