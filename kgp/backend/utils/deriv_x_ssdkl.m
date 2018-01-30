function dx = deriv_x_ssdkl(alpha, P, K, xg, mean, hyp, x, deg, cov, lik, y, opt, xs, p, ng, Dg, N, D, ndcovs)
  % deriv_x   Compute dcovGrid_dx for given parameters.
  %           The function is taken from DKL framework.
  h = 1e-5;
  if P == 1, tP = eye(size(x,2)); else tP = P; end
  xP = x*P';
  xsP = xs*P';
  [M,dM] = covGrid('interp',xg,xP,deg);       % grid interp derivative matrices
  beta = K.mvm(M'*alpha);                         % dP(i,j) = -alpha'*dMij*beta
  dxP = [];
  for i=1:size(tP,2)
    if equi(xg,i)                                              % scaling factor
      wi = max(xg{i})-min(xg{i});
    else
      wi = 1;
    end
    xP(:,i) = xP(:,i) - h;
    [dmi1, fmu1, fs21, ymu1, ys21] = get_deriv_components(hyp, mean, cov, lik, xP, y, opt, xsP, xg, p, ng, Dg, N, D, deg, ndcovs);
    xP(:,i) = xP(:,i) + 2*h;
    [dmi2, fmu2, fs22, ymu2, ys22] = get_deriv_components(hyp, mean, cov, lik, xP, y, opt, xsP, xg, p, ng, Dg, N, D, deg, ndcovs);
    dvi_dx = hyp.ss_strength * sum(ys22 - ys21) / (2*h);
    dmi_dx = (dmi2 - dmi1) / (2*h);  % numerical approximation to the gradient
    xP(:,i) = xP(:,i) - h;
    betai = dmi_dx + dM{i}*beta/wi;

    % add variance

    dxP = [dxP -alpha.*betai+dvi_dx];
  end
  dx = dxP*tP;
end

function eq = equi(xg,i)                        % grid along dim i is equispaced
  ni = size(xg{i},1);
  if ni>1                              % diagnose if data is linearly increasing
    dev = abs(diff(xg{i})-ones(ni-1,1)*(xg{i}(2,:)-xg{i}(1,:)));
    eq = max(dev(:))<1e-9;
  else
    eq = true;
  end
end

function [dmi, fmu, fs2, ymu, ys2] = get_deriv_components(hyp, mean, cov, lik, xP, y, opt, xsP, xg, p, ng, Dg, N, D, deg, ndcovs)
  [K,M] = feval(cov{:}, hyp.cov, xP);    % evaluate covariance mat constituents
  m = feval(mean{:}, hyp.mean, xP);                      % evaluate mean vector
  dmi = m;
  if iscell(lik), lstr = lik{1}; else lstr = lik; end
  if isa(lstr,'function_handle'), lstr = func2str(lstr); end
  if isequal(lstr,'likGauss'), inf = @infGaussLik; else inf = @infLaplace; end
  % inf = @infGaussLik;
  [post nlZ, dnlZ] = inf(hyp, mean, cov, lik, xP, y, opt);

  ns = 0;                       % do nothing per default, 20 is suggested in paper
  if isfield(opt,'pred_var'), ns = max(ceil(abs(opt.pred_var)),20); end
  if ndcovs>0 && nargout>2, ns = max(ns,ndcovs); end  % possibly draw more samples
  Mtal = M'*post.alpha;                         % blow up alpha vector from n to N
  kronmvm = K.kronmvm;
  if ns>0
    s = 3;                                      % Whittle embedding overlap factor
    [V,ee,e] = apxGrid('eigkron',K,xg,s);            % perform eigen-decomposition
    % explained variance on the grid vg=diag(Ku*M'*inv(C)*M*Ku), C=M*Ku*M'+inv(W)
    % relative accuracy r = std(vg_est)/vg_exact = sqrt(2/ns)
    A = sample(V,e,M,post.sW,ns,kronmvm); A = post.L(A);           % a~N(0,inv(C))
    z = K.mvm(M'*A); vg = sum(z.*z,2)/ns;             % z ~ N(0,Ku*M'*inv(C)*M*Ku)
    if ndcovs>0
      dnlZ.covs = - apxGrid('dirder',K,xg,M,post.alpha,post.alpha)/2;
      na = size(A,2);
      for i=1:na                                % compute (E[a'*dK*a] - a'*dK*a)/2
        dnlZ.covs = dnlZ.covs + apxGrid('dirder',K,xg,M,A(:,i),A(:,i))/(2*na);
      end
    end
  else
    vg = zeros(N,1);                                       % no variance explained
  end
  % add fast predictions to post structure, f|y,mu|s2
  post.predict = @(xs) predict(xs,xg,K.mvm(Mtal),vg,hyp,mean,cov,lik,deg);
  [fmu, fs2, ymu, ys2] = post.predict(xsP);
end

% Compute latent and predictive means and variances by grid interpolation.
function [fmu,fs2,ymu,ys2] = predict(xs,xg,Kalpha,vg,hyp,mean,cov,lik,deg)
  Ms = apxGrid('interp',xg,xs,deg);                % obtain interpolation matrix
  xs = apxGrid('idx2dat',xg,xs,deg);                    % deal with index vector
  ms = feval(mean{:},hyp.mean,xs);                         % evaluate prior mean
  fmu = ms + Ms*Kalpha;                 % combine and perform grid interpolation
  if nargout>1
    if norm(vg,1)>1e-10, ve = Ms*vg; else ve = 0; end    % interp grid var expl.
    ks = feval(cov{:},hyp.cov,xs,'diag');              % evaluate prior variance
    fs2 = max(ks-ve,0);              % combine, perform grid interpolation, clip
    if nargout>2, [lp, ymu, ys2] = feval(lik{:},hyp.lik,[],fmu,fs2); end
  end
end
