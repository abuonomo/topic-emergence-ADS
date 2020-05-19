function update(selectedX, selectedY) {
  console.log(selectedX);
  console.log(selectedY);
}


function onTopicClick(d, i) {
  var colorName = 'kmeans_cluster';
  var topic = (+d['topics'] - 1).toString();
  d3.json(page_url + 'get-time-data', {
    method:"POST",
    body: JSON.stringify({
      stem: topic, kmeans_cluster: d[colorName]
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8"
    }
  }).then(
    function(data) {
      timeChart.data(data);
    }
  )
}

function onRectClick(d, i) {
  var colorName = 'kmeans_cluster';
  var topic = d['Term'];
  d3.json(page_url + 'get-kwd-time-data', {
    method:"POST",
    body: JSON.stringify({
      stem: topic, kmeans_cluster: d[colorName]
    }),
    headers: {
      "Content-type": "application/json; charset=UTF-8"
    }
  }).then(
    function(data) {
      timeChart.data(data);
    }
  )
}

function exportToJsonFile(jsonData) {
  let dataStr = JSON.stringify(jsonData);
  let dataUri = 'data:application/json;charset=utf-8,'+ encodeURIComponent(dataStr);

  let exportFileDefaultName = 'data.json';

  let linkElement = document.createElement('a');
  linkElement.setAttribute('href', dataUri);
  linkElement.setAttribute('download', exportFileDefaultName);
  linkElement.click();
}


// Input the data
function scatterChart() {
  var margin = 50;
  var height = 300;
  var width = 500;

  var xName = 'value__mean_change';
  var yName = 'value__cid_ce__normalize_True';
  var labelName = 'stem';
  var colorName = 'kmeans_cluster';
  var sizeVal = "scaled_counts";
  var sizeName = 'count';

  var xValue = function(d) { return d[xName]; };
  var yValue = function(d) { return d[yName]; };
  var cValue = function(d) { return d[colorName];};

  var xScale = d3.scaleLinear();
  var yScale = d3.scaleLinear();
  var xAxis;
  var yAxis;

  var color = d3.scaleOrdinal(d3.schemeCategory10);

  var brush;
  var svg;
  var scatter;
  var xExtent;
  var yExtent;
  var idleTimeout;
  var idleDelay = 350;
  var timeSvg;
  var timeChart =  timeSeriesChart();

  function my(selection) {
    selection.each(function (data, i) {
      svg = d3.select(this);

      var g = svg.append('g')
      .style('transform', 'translate(10%, 10%)');

      scatter = g.append("g")
      .attr("id", "scatterplot")
      .attr("clip-path", "url(#clip)");

      var clip = g.append("defs").append("svg:clipPath")
      .attr("id", "clip")
      .append("svg:rect")
      .attr("width", width+10)
      .attr("height", height+10)
      .attr("x", 0)
      .attr("y", 0)
      .attr("transform", "translate(0,-10)");

      var xAxisEl = g.append("g")
      .attr("class", "x axis")
      .attr('id', "axis--x")
      .attr("transform", "translate(0," + height + ")");

      var yAxisEl = g.append("g")
      .attr("class", "y axis")
      .attr('id', "axis--y");

      var xAxisTitleEl = g.append("text")
      .attr('class', 'axis_title')
      .style("text-anchor", "middle")
      .attr('class', 'axis_title')
      .attr("transform", `translate(${width/2}, ${height + 30})`);

      var yAxisTitleEl = g.append("text")
      .attr('class', 'axis_title')
      .attr("transform", "rotate(-90)")
      .attr("y", -45)
      .attr("x",0 - (height / 2))
      .style("text-anchor", "middle")
      .attr("dy", "1em");

      g.append("text")
      .attr("transform", "translate(10,0)")
      .attr("x", 0)
      .attr("y", -20)
      .attr("class", "title")
      .text(`Keywords Time Series Measures`);

      brush = d3.brush().extent([[0, 0], [width, height]]).on("end", brushended);
      scatter.append("g")
      .attr("class", "brush")
      .call(brush);

      xScale.range([0, width]);
      yScale.range([height, 0]);
      xAxis = d3.axisBottom(xScale).ticks(12);
      yAxis = d3.axisLeft(yScale).ticks(12 * height / width);

      // data dependent
      xExtent = d3.extent(data, function (d) { return d[xName]; });
      yExtent = d3.extent(data, function (d) { return d[yName]; });

      xScale.domain(xExtent).nice();
      yScale.domain(yExtent).nice();

      scatter.selectAll(".dot")
      .data(data)
      .enter().append("circle")
      .attr("class", "dot")
      .attr("pointer-events", "all")
      .attr("clip-path", "url(#clip)")
      .call(tooltip())
      .on("click", onClick)
      .attr("r", function (d) { return d[sizeVal]; })
      .attr("cx", function (d) { return xScale(d[xName]); })
      .attr("cy", function (d) { return yScale(d[yName]); })
      .attr("opacity", 0.5)
      .style("fill", function(d) { return color(cValue(d));});

      xAxisEl
      .call(xAxis);

      xAxisTitleEl
      .text(xName);

      yAxisEl
      .call(yAxis);

      yAxisTitleEl
      .text(yName);
    });
  }

  function brushended() {
    var s = d3.event.selection;
    if (!s) {
      if (!idleTimeout) return idleTimeout = setTimeout(idled, idleDelay);
      xScale.domain(xExtent).nice();
      yScale.domain(yExtent).nice();
    } else {
      xScale.domain([s[0][0], s[1][0]].map(xScale.invert, xScale));
      yScale.domain([s[1][1], s[0][1]].map(yScale.invert, yScale));
      scatter.select(".brush").call(brush.move, null);
    }
    zoom();
  }

  function idled() {
    idleTimeout = null;
  }

  function zoom() {
    var t = scatter.transition().duration(750);
    svg.select("#axis--x").transition(t).call(xAxis);
    svg.select("#axis--y").transition(t).call(yAxis);
    scatter.selectAll("circle").transition(t)
    .attr("cx", function (d) { return xScale(d[xName]); })
    .attr("cy", function (d) { return yScale(d[yName]); });
  }

  function onClick(d, i) {
    d3.json(window.location.pathname + 'get-time-data', {
      method:"POST",
      body: JSON.stringify({
        stem: d[labelName], kmeans_cluster: d[colorName]
      }),
      headers: {
        "Content-type": "application/json; charset=UTF-8"
      }
    }).then(
      function(data) {
        timeChart.data(data);
      }
    );
    var iframeElementx = document.getElementById("pyLDAvis"),
      iframeElementy = (iframeElementx.contentWindow || iframeElementx.contentDocument),
      iframeElementz = iframeElementy.document.body;
    var vizSvg = d3.select(iframeElementz);
    vizSvg.select('#ldavis_el5816652949919526245638977-topic').node().value = (+d[labelName] + 1).toString()
  }

  function tooltip() {
    var tooltipDiv;
    var bodyNode = d3.select('body').node();

    function tooltip(selection){

      selection.on('mouseover.tooltip', function(pD, pI){
        // Clean up lost tooltips
        d3.select('body').selectAll('div.tooltip').remove();
        // Append tooltip
        tooltipDiv = d3.select('body')
        .append('div')
        .attr('class', 'tooltip');
        var absoluteMousePos = d3.mouse(bodyNode);
        tooltipDiv
        .style('left', (absoluteMousePos[0] + 10)+'px')
        .style('top', (absoluteMousePos[1] - 40)+'px');

        var kwd = pD['stem'].toString().replace('<', '&lt;').replace('>', '&gt;');
        var line1 = '<p><strong>' + kwd + '</strong></p>';
        var line2 = `<p>${xName}: ` + pD[xName].toFixed(2) + '</p>';
        var line3 = `<p>${yName}: ` + pD[yName].toFixed(2) + '</p>';
        var line4 = `<p>count: ` + pD[sizeName] + '</p>';
        var line5 = `<p>cluster: ` + pD[colorName] + '</p>';


        tooltipDiv.html(line1 + line2 + line3 + line4 + line5)
      })
      .on('mousemove.tooltip', function(pD, pI){
        // Move tooltip
        var absoluteMousePos = d3.mouse(bodyNode);
        tooltipDiv.style({
          left: (absoluteMousePos[0] + 10)+'px',
          top: (absoluteMousePos[1] - 40)+'px'
        });
      })
      .on('mouseout.tooltip', function(pD, pI){
        // Remove tooltip
        tooltipDiv.remove();
      });

    }

    tooltip.attr = function(_x){
      if (!arguments.length) return attrs;
      attrs = _x;
      return this;
    };

    tooltip.style = function(_x){
      if (!arguments.length) return styles;
      styles = _x;
      return this;
    };

    return tooltip;
  }

  // getter-setters
  my.data = function (value) {
    if (arguments.length === 0) return data;
    data = value;
    if (typeof updateData === 'function') updateData();
    return my;
  };

  my.height = function (value) {
    if (arguments.length === 0) return height;
    height = value;
    return my;
  };

  my.width = function (value) {
    if (arguments.length === 0) return width;
    width = value;
    return my;
  };

  my.xName = function (value) {
    if (arguments.length === 0) return xName;
    xName = value;
    return my;
  };

  my.yName = function (value) {
    if (arguments.length === 0) return yName;
    yName = value;
    return my;
  };

  my.colorName = function (value) {
    if (arguments.length === 0) return colorName;
    colorName = value;
    return my;
  };

  my.sizeVal = function (value) {
    if (arguments.length === 0) return sizeVal;
    sizeVal = value;
    return my;
  };

  my.sizeName = function (value) {
    if (arguments.length === 0) return sizeName;
    sizeName = value;
    return my;
  };

  my.labelName = function (value) {
    if (arguments.length === 0) return labelName;
    labelName = value;
    return my;
  };

  my.timeSvg = function(value) {
    if (!arguments.length) return timeSvg;
    timeSvg = value;
    return my;
  };

  my.timeChart = function(value) {
    if (!arguments.length) return timeChart;
    timeChart = value;
    return my;
  };

  my.color = function(value) {
    if (!arguments.length) return color;
    color = value;
    return my;
  };

  my.xScale = function(value) {
    if (!arguments.length) return xScale;
    xScale = value;
    return my;
  };

  return my;
}

function timeSeriesChart() {

  var data = [];
  var height = 400;
  var width = 800;

  var xName = 'year';
  var yName = 'count';
  var colorName = 'kmeans_cluster';
  var cValue = function(d) { return d[colorName];}; // Won't change with colorName;
  var color = d3.scaleOrdinal(d3.schemeCategory10);

  var parseTime = d3.timeParse("%Y");
  var xTime = d3.scaleTime();
  var yTime = d3.scaleLinear();
  var formatData;

  function my(selection) {
    selection.each(function() {
      svg = d3.select(this);
      const t = svg.transition().duration(750);

      xTime.range([0, width]);
      yTime.range([height, 0]);

      var tg = svg.append('g')
      .style('transform', 'translate(10%, 10%)');

      var g = tg.append("g")
      .attr("id", "timeSeries");

      var xAxisEl = g.append("g")
      .attr('id', 'xAxis');

      var yAxisEl = g.append("g")
      .attr('id', 'yAxis');

      formatData = function(d) {
        d[xName] = parseTime(d[xName]);
        d[yName] = +d[yName];
      };

      updateData = function() {
        data.forEach(formatData);
        _updateData();
      };

      _updateData = function() {
        xTime.domain(d3.extent(data, function(d) { return d[xName]; }));
        yTime.domain([0, d3.max(data, function(d) { return d[yName]; })]);

        var line = d3.line()
        .x(function(d) { return xTime(d[xName]); })
        .y(function(d) { return yTime(d[yName]); });

        var area = d3.area()
        .x(function(d) { return xTime(d[xName]); })
        .y0(height)
        .y1(function(d) { return yTime(d[yName]); });

        xAxisEl
        .attr("transform", "translate(0," + height + ")")
        .call(d3.axisBottom(xTime).ticks(d3.timeYear.every(1)))
        .selectAll("text")
        .join('text')
        .attr("y", 10)
        .attr("x", -10)
        .attr("dy", ".35em")
        .attr("transform", "rotate(0)")
        .style("text-anchor", "start");

        yAxis = d3.axisLeft(yTime).tickFormat(function(d){
          return d;
        }).ticks(5);

        yAxisEl
        .transition(t)
        .call(yAxis);

        var tmpColor = color(cValue(data[0])); // All the same so get first

        g.selectAll(".area")
        .data([data])
        .join(
          enter => enter.append('path')
          .attr("d", area)
          .style("stroke", tmpColor),
          update => update
          .call(update => update.transition(t)
            .attr('d', area)
            .style('fill', tmpColor)
          )
        )
        .attr("class", "area");

        // add the valueline path.
        g.selectAll(".line")
        .data([data])
        .join(
          enter => enter.append('path')
          .attr("d", line)
          .style("stroke", tmpColor),
          update => update
          .call(update => update.transition(t)
            .attr('d', line)
            .style('stroke', tmpColor)
          )
        )
        .attr("class", "line");

        g.selectAll('circle')
        .data(data)
        .join(
          enter => enter.append("circle")
          .attr("r", 4)
          .style('opacity', 0.5)
          .style('fill', tmpColor)
          .attr("cx", function(d) { return xTime(d[xName]); })
          .attr("cy", function(d) { return yTime(d[yName]); }),
          update => update
          .call(update => update.transition(t)
            .style('fill', tmpColor)
            .attr("cx", function(d) { return xTime(d[xName]); })
            .attr("cy", function(d) { return yTime(d[yName]); })
          ),
        );

        g.selectAll('.title')
        .data([data[0]['stem']])
        .join(
          enter => enter.append('text').text(function (d) {return `\"${d}\" year versus count`}),
          update => update.text(function (d) {return `\"${d}\" year versus count`}),
        )
        .attr("transform", "translate(10,0)")
        .attr("x", 0)
        .attr("y", -20)
        .attr("class", "title");
      };

      updateData();

      g.append("text")
      .style("text-anchor", "middle")
      .attr('class', 'axis_title')
      .attr("transform", `translate(${width/2}, ${height + 30})`)
      .text(xName);

      g.append("text")
      .attr("transform", "rotate(-90)")
      .attr('class', 'axis_title')
      .style("text-anchor", "middle")
      .attr("y", -45)
      .attr("x",0 - (height / 2))
      .attr("dy", "1em")
      .text(yName);

    })
  }

  // getter-setters
  my.data = function (value) {
    if (arguments.length === 0) return data;
    data = value;
    if (typeof updateData === 'function') updateData();
    return my;
  };

  my.height = function (value) {
    if (arguments.length === 0) return height;
    height = value;
    return my;
  };

  my.width = function (value) {
    if (arguments.length === 0) return width;
    width = value;
    return my;
  };

  my.xName = function (value) {
    if (arguments.length === 0) return xName;
    xName = value;
    return my;
  };

  my.yName = function (value) {
    if (arguments.length === 0) return yName;
    yName = value;
    if (typeof _updateData === 'function') _updateData();
    return my;
  };

  my.colorName = function (value) {
    if (arguments.length === 0) return colorName;
    colorName = value;
    return my;
  };

  my.color = function(value) {
    if (!arguments.length) return color;
    color = value;
    return my;
  };

  return my;
}
$(document).ready(function() {
  $('.selector').select2();
});

$(".selector").select2({
  width: 'resolve' // need to override the changed default
});
