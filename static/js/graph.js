Highcharts.chart('container', {
    chart: {
        type: 'column'
    },
    title: {
        text: '자세교정AI 경고 알림수'
    },
    xAxis: {
        categories: ['2023-01-15', '2023-01-16', '2023-01-17', '2023-01-18', 'Today']
    },
    series: [{
        name: '거북목',
        data: [107, 31, 635, 203, 2]
    }, {
        name: '다리교정',
        data: [133, 156, 947, 408, 6]
    }, {
        name: '어깨교정',
        data: [1052, 954, 4250, 740, 38]
    }, {
        name: '입술뜯는습관',
        data: [1052, 954, 4250, 740, 38]
  }]
});